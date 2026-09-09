from __future__ import annotations

import numpy as np
import pysaliency as ps
import torch
from utils.saliency_utils import get_saliency_map_transalnet
from models.TranSalNet.TranSalNet_Dense import TranSalNet

# --- pysaliency 需要 SaliencyMapModel，而不是直接 array ---
class OneMapModel(ps.SaliencyMapModel):
    def __init__(self, saliency_map: np.ndarray, stimulus: np.ndarray):
        super().__init__()
        self._map = np.asarray(saliency_map, dtype=np.float32)
        self._stim_id = id(stimulus)

    def _saliency_map(self, stimulus):
        if id(stimulus) != self._stim_id:
            raise ValueError("Unexpected stimulus object.")
        return self._map


class TwoMapEvalContext:
    """把 stimulus/stimuli/pred_model/gt_model 放一起，方便复用。"""
    def __init__(self, s_pred: np.ndarray, s_target: np.ndarray):
        s_pred = np.asarray(s_pred)
        s_target = np.asarray(s_target)
        assert s_pred.shape == s_target.shape and s_pred.ndim == 2, \
            f"s_pred and s_target must be same [H,W], got {s_pred.shape} vs {s_target.shape}"

        self.s_pred = s_pred
        self.s_target = s_target

        # stimuli 只是索引载体；内容不重要，但必须是 numpy 图像对象
        self.stimulus = np.zeros(s_pred.shape, dtype=np.uint8)
        self.stimuli = ps.Stimuli([self.stimulus])

        self.pred_model = OneMapModel(s_pred, self.stimulus)
        self.gt_model = OneMapModel(s_target, self.stimulus)


# =========================
# Base class（你要的 basedsaliency）
# =========================
class BasedSaliencyMetric:
    """
    输入：两张 saliency map（s_pred, s_target），shape [H,W]
    输出：score(float), stats(dict)
    """
 

    def __init__(self, 
        *,
        eps: float = 1e-20,
        mode: Literal["KL", "CC", "SIM"] = "KL",
        ):
        self.mode = mode  # 子类填：'KL' / 'CC' / 'SIM'
        self.eps = float(eps)
        model = TranSalNet()
        model.load_state_dict(torch.load('pretrained/TranSalNet_Dense.pth'))
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device) 
        self.model.eval()
        self.metric_mode = 'based_saliency'
    
    def _get_saliency(self,img):
        return get_saliency_map_transalnet(self.model,img)

    def __call__(self, pred: np.ndarray, target: np.ndarray):
        def to_img(t):
            if isinstance(t,torch.Tensor):
                t = (t[0].cpu().numpy().transpose(1,2,0) * 255.0).clip(0,255).astype(np.uint8)
            return t

        pred = to_img(pred)
        target = to_img(target)

        
        s_pred = self._get_saliency(pred[:,:,::-1])
        s_target = self._get_saliency(target[:,:,::-1])

        ctx = TwoMapEvalContext(s_pred, s_target)
        mode = (self.mode or "").upper()

        if mode == "KL":
            val = ctx.pred_model.image_based_kl_divergence(
                ctx.stimuli, ctx.gt_model, minimum_value=self.eps
            )
        elif mode == "CC":
            val = ctx.pred_model.CC(ctx.stimuli, ctx.gt_model)
        elif mode == "SIM":
            val = ctx.pred_model.SIM(ctx.stimuli, ctx.gt_model)
        else:
            raise ValueError(f"Unknown mode={self.mode}. Expected KL/CC/SIM")

        stats = {"metric": mode}
        return torch.tensor(float(val)), stats

    def visualize(self, s_pred: np.ndarray, s_target: np.ndarray, *args, **kwargs):
        # 可选：你也可以学 SDAMetric 那样拼 panel
        return None


# =========================
# 三个子类（你要的“3个类继承 basedsaliency”）
# =========================
class BasedSaliency_KL(BasedSaliencyMetric):
    def __init__(self):
        super().__init__(mode='KL')


class BasedSaliency_CC(BasedSaliencyMetric):
    def __init__(self):
        super().__init__(mode='CC')


class BasedSaliency_SIM(BasedSaliencyMetric):
    def __init__(self):
        super().__init__(mode='SIM')
