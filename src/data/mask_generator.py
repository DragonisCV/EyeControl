"""
Online Mask Augmentor (v0)
- Designed for PyTorch Dataset training-time augmentation.
- Takes intent dict loaded from .intent.npz:
    {
      "coarse_region_pos": np.uint8 [H,W] (0/1),
      "peaks_pos_xy": np.int16 [K,2] (x,y),
      "peaks_pos_score": np.float32 [K],
      # optional:
      "skeleton_pos": np.uint8 [H,W] (0/1),
    }
- Returns soft mask float32 [H,W] in [0,1], plus optional debug info.

Key design:
- Decouple "intent" (coarse region + peaks) from "tool shape" (click/stroke/region).
- Online randomization: brush size, feather, jitter, number of points, etc.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, List

import numpy as np
import torch
import torch.nn.functional as F
from src.data.mask_utils import _as_np,_clamp01,_log_uniform,_choice_weighted,_meshgrid_xy,_randint,_ensure_hw
from src.data.mask_utils import _torch_gaussian_blur_2d,_torch_morph_dilate,_torch_morph_erode,get_cc_labels,assign_peaks_to_cc
import os
import cv2
from PIL import Image
# -------------------------
# Augmentor
# -------------------------

class OnlineMaskAugmentor:
    """
    Online mask augmentation. Parameters are provided via a dict (YAML-friendly).
    """

    def __init__(self, cfg: Dict[str, Any], seed: Optional[int] = None):
        self.cfg = cfg
        self.rng = np.random.default_rng(seed)

        # tool probabilities
        self.p_click = float(cfg.get("p_click", 0.5))
        self.p_stroke = float(cfg.get("p_stroke", 0.3))
        self.p_region = float(cfg.get("p_region", 0.2))
        s = self.p_click + self.p_stroke + self.p_region
        if s <= 1e-9:
            self.p_click, self.p_stroke, self.p_region = 1.0, 0.0, 0.0
        else:
            self.p_click /= s
            self.p_stroke /= s
            self.p_region /= s

        # click params
        self.click_n_choices = cfg.get("click_n_choices", [1, 2, 3, 4])
        self.click_n_probs = cfg.get("click_n_probs", [0.5, 0.3, 0.15, 0.05])
        self.click_jitter_ratio = float(cfg.get("click_jitter_ratio", 0.01))  # *max_side
        self.click_r_min = float(cfg.get("click_r_min", 0.006))               # *max_side
        self.click_r_max = float(cfg.get("click_r_max", 0.030))               # *max_side
        self.click_sigma_mul_min = float(cfg.get("click_sigma_mul_min", 0.6))
        self.click_sigma_mul_max = float(cfg.get("click_sigma_mul_max", 1.5))
        self.click_score_gamma = float(cfg.get("click_score_gamma", 1.0))
        self.click_cluster_near_ratio = float(cfg.get("click_cluster_near_ratio", 0.06))  # *max_side

        # stroke params
        self.stroke_use_points_prob = float(cfg.get("stroke_use_points_prob", 0.8))  # use peaks to connect, else skeleton
        self.stroke_n_choices = cfg.get("stroke_n_choices", [2, 3, 4])
        self.stroke_n_probs = cfg.get("stroke_n_probs", [0.6, 0.3, 0.1])
        self.stroke_w_min = float(cfg.get("stroke_w_min", 0.008))  # *max_side
        self.stroke_w_max = float(cfg.get("stroke_w_max", 0.040))  # *max_side
        self.stroke_jitter_ratio = float(cfg.get("stroke_jitter_ratio", 0.01))
        self.stroke_sigma_mul_min = float(cfg.get("stroke_sigma_mul_min", 0.4))
        self.stroke_sigma_mul_max = float(cfg.get("stroke_sigma_mul_max", 1.2))

        # region params
        self.region_r_min = float(cfg.get("region_r_min", 0.006))  # *max_side
        self.region_r_max = float(cfg.get("region_r_max", 0.040))  # *max_side
        self.region_erode_prob = float(cfg.get("region_erode_prob", 0.35))
        self.region_dropout_prob = float(cfg.get("region_dropout_prob", 0.15))
        self.region_dropout_block_ratio = float(cfg.get("region_dropout_block_ratio", 0.03))  # *max_side
        self.region_sigma_min = float(cfg.get("region_sigma_min", 0.6))  # px
        self.region_sigma_max = float(cfg.get("region_sigma_max", 3.0))  # px

        # output / shared
        self.out_soft = bool(cfg.get("out_soft", True))
        self.feather_prob = float(cfg.get("feather_prob", 0.9))
        self.soft_composite = str(cfg.get("soft_composite", "noisy_or"))  # "max" or "noisy_or"
        self.device = cfg.get("device", None)  # if None, stay numpy; if set, use torch ops on this device
        self.return_debug = bool(cfg.get("return_debug", False))

    # -------------------------
    # Public API
    # -------------------------

    def __call__(self, intent: Dict[str, Any], polarity: str = "pos") -> Dict[str, Any]:
        """
        Returns dict:
          {
            "mask": torch.FloatTensor [H,W] in [0,1],
            "tool": "click"/"stroke"/"region",
            "debug": {...} optional
          }
        """
        coarse = _as_np(intent.get(f"coarse_region_{polarity}", None))
        peaks_xy = _as_np(intent.get(f"peaks_{polarity}_xy", None))
        peaks_sc = _as_np(intent.get(f"peaks_{polarity}_score", None))
        skel = _as_np(intent.get(f"skeleton_{polarity}", None))

        if coarse is None:
            raise KeyError(f"intent missing coarse_region_{polarity}")

        H, W = int(coarse.shape[0]), int(coarse.shape[1])
        max_side = max(H, W)

        # sanitize peaks
        if peaks_xy is None or len(peaks_xy) == 0:
            peaks_xy = np.zeros((0, 2), dtype=np.int16)
            peaks_sc = np.zeros((0,), dtype=np.float32)
        else:
            peaks_xy = np.asarray(peaks_xy, dtype=np.int32).reshape(-1, 2)
            peaks_sc = np.asarray(peaks_sc, dtype=np.float32).reshape(-1)
            if len(peaks_sc) != len(peaks_xy):
                peaks_sc = np.ones((len(peaks_xy),), dtype=np.float32)

        # pick tool
        u = float(self.rng.random())
        if u < self.p_click:
            tool = "click"
            mask_np, dbg = self._gen_click(H, W, peaks_xy, peaks_sc, coarse,max_side)
        elif u < self.p_click + self.p_stroke:
            tool = "stroke"
            mask_np, dbg = self._gen_stroke(H, W, peaks_xy, peaks_sc, coarse, skel, max_side)
        else:
            tool = "region"
            mask_np, dbg = self._gen_region(H, W, coarse, max_side)

        # optionally feather (soften)
        if self.out_soft and (float(self.rng.random()) < self.feather_prob):
            # choose sigma depending on tool
            if tool == "click":
                # already soft; slight extra blur is ok
                sigma = _log_uniform(self.rng, 0.4, 2.0)
            elif tool == "stroke":
                sigma = _log_uniform(self.rng, 0.6, 3.0)
            else:
                sigma = _log_uniform(self.rng, self.region_sigma_min, self.region_sigma_max)
            mask_np = self._blur_soft(mask_np, sigma=sigma)

        mask_np = _clamp01(mask_np).astype(np.float32)

        out = {"mask": torch.from_numpy(mask_np), "tool": tool}
        if self.return_debug:
            out["debug"] = dbg
        return out

    # -------------------------
    # Generators
    # -------------------------

    def _gen_click(
        self,
        H: int,
        W: int,
        peaks_xy: np.ndarray,
        peaks_sc: np.ndarray,
        coarse: np.ndarray,          # <-- 新增：coarse region (0/1 or [0,1])
        max_side: int
    ) -> Tuple[np.ndarray, Dict]:
        dbg = {"mode": "click"}

        # sample N clicks
        n = int(self.rng.choice(self.click_n_choices, p=np.array(self.click_n_probs) / np.sum(self.click_n_probs)))
        n = max(1, n)

        # if no peaks, fallback: random points xxxxxx update zero masks.
        if len(peaks_xy) == 0:
            pts = []
            # for _ in range(n):
            #     x = _randint(self.rng, 0, W - 1)
            #     y = _randint(self.rng, 0, H - 1)
            #     pts.append((x, y))
            pts = np.array(pts, dtype=np.int32)

        else:
            # weighted sampling by score^gamma
            w = np.power(np.maximum(peaks_sc, 1e-6), self.click_score_gamma).astype(np.float64)

            # ---- Click-only rule: multi-click prefers different coarse connected components ----
            # If coarse is invalid, fallback to your original behavior
            use_cc_rule = (coarse is not None) and (coarse.shape[0] == H) and (coarse.shape[1] == W) and (n > 1)

            if use_cc_rule:
                coarse_u8 = (coarse > 0).astype(np.uint8)

                # connected components
                num, labels, stats, _ = cv2.connectedComponentsWithStats(coarse_u8, connectivity=8)

                # assign each peak to a component id (0 = background)
                xs = np.clip(peaks_xy[:, 0].astype(np.int32), 0, W - 1)
                ys = np.clip(peaks_xy[:, 1].astype(np.int32), 0, H - 1)
                peak_comp = labels[ys, xs].astype(np.int32)

                # group peak indices by component (ignore background)
                comp_to_idxs = {}
                for i, cid in enumerate(peak_comp.tolist()):
                    if cid <= 0:
                        continue
                    comp_to_idxs.setdefault(cid, []).append(i)

                # if no peak falls inside coarse, fallback to global weighted
                if len(comp_to_idxs) == 0:
                    pick = _choice_weighted(self.rng, np.arange(len(peaks_xy)), w, k=n, replace=False)
                    pts = peaks_xy[pick].astype(np.int32)

                else:
                    # (1) first pass: pick 1 point per distinct component
                    comp_ids = list(comp_to_idxs.keys())

                    # component weights: sum(w in comp) * sqrt(area)
                    comp_w = []
                    for cid in comp_ids:
                        idxs = np.array(comp_to_idxs[cid], dtype=np.int32)
                        sw = float(np.sum(w[idxs]))
                        area = float(stats[cid, cv2.CC_STAT_AREA]) if cid < stats.shape[0] else 1.0
                        comp_w.append(sw * np.sqrt(max(area, 1.0)))
                    comp_w = np.array(comp_w, dtype=np.float64)
                    comp_w = comp_w / (comp_w.sum() + 1e-12)

                    pts_list = []
                    picked_comps = []

                    remaining = comp_ids.copy()
                    remaining_w = comp_w.copy()

                    k1 = min(n, len(remaining))
                    for _ in range(k1):
                        remaining_w = remaining_w / (remaining_w.sum() + 1e-12)
                        j = int(self.rng.choice(len(remaining), p=remaining_w))
                        cid = int(remaining.pop(j))
                        remaining_w = np.delete(remaining_w, j)

                        idxs = np.array(comp_to_idxs[cid], dtype=np.int32)
                        ww = w[idxs].astype(np.float64)
                        ww = ww / (ww.sum() + 1e-12)
                        pi = int(self.rng.choice(len(idxs), p=ww))
                        pts_list.append(peaks_xy[idxs[pi]].tolist())
                        picked_comps.append(cid)

                    # (2) supplement if still need more points: avoid too-close clicks
                    min_dist_ratio = getattr(self, "click_min_dist_ratio", 0.06)
                    min_dist = float(min_dist_ratio) * max_side
                    min_dist2 = min_dist * min_dist

                    if len(pts_list) < n:
                        order = np.argsort(-w)  # global preference by score
                        for ii in order.tolist():
                            if len(pts_list) >= n:
                                break
                            p = peaks_xy[ii].astype(np.float32)
                            ok = True
                            for q in pts_list:
                                dx = p[0] - q[0]
                                dy = p[1] - q[1]
                                if (dx * dx + dy * dy) < min_dist2:
                                    ok = False
                                    break
                            if ok:
                                pts_list.append(peaks_xy[ii].tolist())

                    pts = np.array(pts_list, dtype=np.int32)
                    dbg.update({"picked_comps": picked_comps, "min_dist_ratio": float(min_dist_ratio)})

            else:
                # ---- original behavior (kept) ----
                first = _choice_weighted(self.rng, np.arange(len(peaks_xy)), w, k=1, replace=False)[0]
                pts = [peaks_xy[first].tolist()]

                near_r = self.click_cluster_near_ratio * max_side
                near_r2 = near_r * near_r
                for _ in range(n - 1):
                    base = np.array(pts[-1], dtype=np.float32)
                    d2 = np.sum((peaks_xy.astype(np.float32) - base[None, :]) ** 2, axis=1)
                    near_idx = np.where(d2 <= near_r2)[0]
                    if len(near_idx) == 0:
                        pick = _choice_weighted(self.rng, np.arange(len(peaks_xy)), w, k=1, replace=False)[0]
                        pts.append(peaks_xy[pick].tolist())
                    else:
                        pick = _choice_weighted(self.rng, near_idx, w[near_idx], k=1, replace=False)[0]
                        pts.append(peaks_xy[pick].tolist())
                pts = np.array(pts, dtype=np.int32)

        # jitter points
        jitter = int(round(self.click_jitter_ratio * max_side))
        if jitter > 0:
            pts[:, 0] = np.clip(pts[:, 0] + self.rng.integers(-jitter, jitter + 1, size=len(pts)), 0, W - 1)
            pts[:, 1] = np.clip(pts[:, 1] + self.rng.integers(-jitter, jitter + 1, size=len(pts)), 0, H - 1)

        # render soft clicks
        X, Y = _meshgrid_xy(H, W)
        mask = np.zeros((H, W), dtype=np.float32)

        for (x, y) in pts.tolist():
            r = _log_uniform(self.rng, self.click_r_min, self.click_r_max) * max_side
            sigma_mul = float(self.rng.uniform(self.click_sigma_mul_min, self.click_sigma_mul_max))
            sigma = max(1.0, r * sigma_mul)
            dx = X - float(x)
            dy = Y - float(y)
            g = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma)).astype(np.float32)

            if self.soft_composite == "max":
                mask = np.maximum(mask, g)
            else:
                mask = 1.0 - (1.0 - mask) * (1.0 - g)

        dbg.update({"n": int(len(pts)), "pts": pts.tolist()})
        return mask, dbg


    def _gen_stroke(
        self,
        H: int,
        W: int,
        peaks_xy: np.ndarray,
        peaks_sc: np.ndarray,
        coarse: np.ndarray,
        skel: Optional[np.ndarray],  # 不再依赖它（可留接口）
        max_side: int
    ):
        import cv2
        dbg = {"mode": "stroke"}

        coarse_u8 = (coarse > 0).astype(np.uint8)
        num, labels, stats = get_cc_labels(coarse_u8)

        # stroke width
        stroke_w = int(round(_log_uniform(self.rng, self.stroke_w_min, self.stroke_w_max) * max_side))
        stroke_w = max(1, stroke_w)

        # --- choose a component (exclude background id=0) ---
        # component weights: area (or area * sumpeakscore)
        comp_ids = np.arange(1, num, dtype=np.int32)
        if len(comp_ids) == 0:
            # no coarse, fallback to empty
            return np.zeros((H, W), np.float32), {"mode": "stroke", "reason": "no_component"}

        # map peaks -> component
        peak_comp = assign_peaks_to_cc(peaks_xy, labels)
        # area weight
        areas = stats[1:, cv2.CC_STAT_AREA].astype(np.float32)  # for comp 1..num-1
        w_area = areas / (areas.sum() + 1e-6)

        # optional: peak-score boost per component
        w_peak = np.zeros((num,), dtype=np.float32)
        if len(peaks_xy) > 0:
            for cid, sc in zip(peak_comp.tolist(), peaks_sc.tolist()):
                if cid > 0:
                    w_peak[cid] += float(sc)
        w_peak = w_peak[1:]  # align to comp 1..num-1
        if w_peak.sum() > 1e-6:
            w_peak = w_peak / (w_peak.sum() + 1e-6)
            comp_w = 0.6 * w_area + 0.4 * w_peak
        else:
            comp_w = w_area

        comp_w = comp_w / (comp_w.sum() + 1e-6)
        chosen_comp = int(self.rng.choice(comp_ids, p=comp_w))

        dbg["chosen_comp"] = chosen_comp
        dbg["stroke_w"] = int(stroke_w)

        # mask of chosen component
        comp_mask = (labels == chosen_comp).astype(np.uint8)

        # peaks within chosen component
        idx_in = np.where(peak_comp == chosen_comp)[0]
        use_points = (len(idx_in) >= 2) and (float(self.rng.random()) < self.stroke_use_points_prob)

        canvas = np.zeros((H, W), dtype=np.float32)

        # sample n points from peaks in this component
        n = int(self.rng.choice(self.stroke_n_choices, p=np.array(self.stroke_n_probs) / np.sum(self.stroke_n_probs)))
        n = max(2, min(n, len(idx_in)))
        local_xy = peaks_xy[idx_in].astype(np.int32)
        local_sc = peaks_sc[idx_in].astype(np.float32)
        w = np.power(np.maximum(local_sc, 1e-6), 1.0)
        pick_local = _choice_weighted(self.rng, np.arange(len(local_xy)), w, k=n, replace=False)
        pts = local_xy[pick_local].astype(np.int32)
        # order by nearest-neighbor (tour)
        ordered = [pts[0]]
        used = {0}
        for _ in range(1, len(pts)):
            last = ordered[-1].astype(np.float32)
            d2 = np.sum((pts.astype(np.float32) - last[None, :]) ** 2, axis=1)
            best = None
            bestv = 1e18
            for j in range(len(pts)):
                if j in used:
                    continue
                if float(d2[j]) < bestv:
                    bestv = float(d2[j])
                    best = j
            used.add(best)
            ordered.append(pts[best])
        pts = np.stack(ordered, axis=0)
        # if any adjacent segment too long -> fallback to peak-centered scribble
        max_link = 0.22 * max_side   # 建议做成 cfg: stroke_max_link_dist_ratio
        too_long = False
        for a, b in zip(pts[:-1], pts[1:]):
            if np.linalg.norm((a - b).astype(np.float32)) > max_link:
                too_long = True
                break
        if too_long:
            use_points = False  # 走 scribble 分支

        if use_points:
            # jitter but keep inside component (简单做：抖动后如果出界就不抖)
            jitter = int(round(self.stroke_jitter_ratio * max_side))
            if jitter > 0:
                pts2 = pts.copy()
                pts2[:, 0] = np.clip(pts2[:, 0] + self.rng.integers(-jitter, jitter + 1, size=len(pts2)), 0, W - 1)
                pts2[:, 1] = np.clip(pts2[:, 1] + self.rng.integers(-jitter, jitter + 1, size=len(pts2)), 0, H - 1)
                ok = comp_mask[pts2[:, 1], pts2[:, 0]] > 0
                pts[ok] = pts2[ok]



            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=False, color=1.0,
                        thickness=stroke_w, lineType=cv2.LINE_AA)
            # region clip
            canvas = canvas * comp_mask.astype(np.float32)


            dbg.update({"use_points": True, "n": int(len(pts)), "pts": pts.tolist()})

        else:
            # --------- scribble random-walk inside component (more natural than skeleton) ---------
            # start point: if have a peak in this comp, start there, else random pixel in comp
            if len(idx_in) >= 1:
                start = peaks_xy[int(idx_in[self.rng.integers(0, len(idx_in))])].astype(np.int32)
                x, y = int(start[0]), int(start[1])
            else:
                ys, xs = np.where(comp_mask > 0)
                ridx = int(self.rng.integers(0, len(xs)))
                x, y = int(xs[ridx]), int(ys[ridx])

            # walk length proportional to component size (cap it)
            area = int(comp_mask.sum())
            L = int(np.clip(0.15 * math.sqrt(area), 20, 180))  # 经验值：更像“涂抹”
            step = max(1, int(round(0.008 * max_side)))       # 步长
            step = min(step, 6)

            # direction with inertia
            ang = float(self.rng.uniform(0, 2 * math.pi))
            pts = [(x, y)]
            for _ in range(L):
                ang = 0.85 * ang + 0.15 * float(self.rng.uniform(0, 2 * math.pi))  # 惯性+噪声
                nx = int(round(x + step * math.cos(ang)))
                ny = int(round(y + step * math.sin(ang)))
                nx = max(0, min(W - 1, nx))
                ny = max(0, min(H - 1, ny))

                if comp_mask[ny, nx] == 0:
                    # bounce: try a few random directions
                    found = False
                    for _try in range(6):
                        ang2 = float(self.rng.uniform(0, 2 * math.pi))
                        tx = int(round(x + step * math.cos(ang2)))
                        ty = int(round(y + step * math.sin(ang2)))
                        tx = max(0, min(W - 1, tx))
                        ty = max(0, min(H - 1, ty))
                        if comp_mask[ty, tx] > 0:
                            nx, ny = tx, ty
                            ang = ang2
                            found = True
                            break
                    if not found:
                        break

                x, y = nx, ny
                pts.append((x, y))

            pts = np.array(pts, dtype=np.int32)
            cv2.polylines(canvas, [pts.reshape(-1, 1, 2)], isClosed=False, color=1.0,
                        thickness=stroke_w, lineType=cv2.LINE_AA)

            dbg.update({"use_points": False, "scribble_len": int(len(pts)), "start": (int(pts[0,0]), int(pts[0,1]))})

        # soften stroke
        sigma_mul = float(self.rng.uniform(self.stroke_sigma_mul_min, self.stroke_sigma_mul_max))
        sigma = max(1.0, sigma_mul * stroke_w)
        mask = self._blur_soft(canvas, sigma=sigma)

        return mask, dbg
    def _gen_region(self, H: int, W: int, coarse: np.ndarray, max_side: int) -> Tuple[np.ndarray, Dict]:
        dbg = {"mode": "region"}
        base = (coarse > 0).astype(np.float32)

        # sample radius
        r = int(round(_log_uniform(self.rng, self.region_r_min, self.region_r_max) * max_side))
        r = max(1, r)

        t = torch.from_numpy(base[None, None, :, :])  # [1,1,H,W]
        # dilate by default, sometimes erode
        if float(self.rng.random()) < self.region_erode_prob:
            t = _torch_morph_erode((t > 0.5).float(), radius=r)
            dbg["op"] = "erode"
        else:
            t = _torch_morph_dilate((t > 0.5).float(), radius=r)
            dbg["op"] = "dilate"

        # optional dropout blocks to simulate imperfect painting
        if float(self.rng.random()) < self.region_dropout_prob:
            block = int(round(self.region_dropout_block_ratio * max_side))
            block = max(4, block)
            for _ in range(_randint(self.rng, 1, 4)):
                x0 = _randint(self.rng, 0, max(0, W - block))
                y0 = _randint(self.rng, 0, max(0, H - block))
                t[:, :, y0:y0 + block, x0:x0 + block] *= float(self.rng.uniform(0.0, 0.5))
            dbg["dropout"] = True
        else:
            dbg["dropout"] = False

        mask = t[0, 0].numpy().astype(np.float32)

        # convert to soft-ish region by slight blur (later global feather may apply too)
        if self.out_soft:
            sigma = _log_uniform(self.rng, self.region_sigma_min, self.region_sigma_max)
            mask = self._blur_soft(mask, sigma=sigma)

        dbg.update({"r": int(r)})
        return mask, dbg

    # -------------------------
    # Soft blur (numpy or torch)
    # -------------------------

    def _blur_soft(self, mask_np: np.ndarray, sigma: float) -> np.ndarray:
        sigma = float(sigma)
        if sigma <= 1e-6:
            return mask_np

        # Use torch blur for deterministic-ish behavior and no OpenCV dependency.
        t = torch.from_numpy(mask_np[None, None, :, :].astype(np.float32))
        t = _torch_gaussian_blur_2d(t, sigma=sigma)
        return t[0, 0].numpy().astype(np.float32)
