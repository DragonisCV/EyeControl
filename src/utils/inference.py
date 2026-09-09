
from src.archs.eyecontrol_transformer import FluxAttention, FluxAttentionZero
import matplotlib.pyplot as plt
import torch
from PIL import Image
from tqdm import tqdm
import numpy as np
import cv2
import os
def save_combined_attention_maps(
    combined_list,
    module_names,
    titles=None,
    save_path=None
):
    """
    combined_list: List[np.ndarray], 每个元素是一个 combined attention map
    module_names: List[str], y 轴模块名
    titles: List[str] or None, 每个子图的标题
    """
    num_maps = len(combined_list)
    num_modules = len(module_names)

    # 动态计算高度
    fig_height = num_modules * 4

    # 横向子图
    fig, axes = plt.subplots(
        1, num_maps,
        figsize=(8 * num_maps, fig_height)
    )

    # 当只有一个 map 时，axes 不是 list
    if num_maps == 1:
        axes = [axes]

    # 计算单个模块高度（假设所有 combined 形状一致）
    single_map_h = combined_list[0].shape[0] // num_modules
    tick_positions = (
        np.arange(num_modules) * single_map_h + (single_map_h / 2)
    )

    for i, (combined, ax) in enumerate(zip(combined_list, axes)):
        ax.imshow(combined, cmap='viridis', aspect='equal')

        if titles is not None:
            ax.set_title(titles[i], fontsize=15)
        else:
            ax.set_title(f'Combined Attention Map {i}', fontsize=15)

        ax.set_yticks(tick_positions)
        ax.set_yticklabels(module_names, fontsize=10)
        ax.set_xticks([])

    plt.tight_layout()

    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        fig.savefig(save_path, bbox_inches='tight', dpi=200)

    plt.close(fig)


def load_image_safely(image_path, size):
    try:
        image = Image.open(image_path).convert("RGB")
        return image
    except Exception as e:
        print("file error: "+image_path)
        with open("failed_images.txt", "a") as f:
            f.write(f"{image_path}\n")
        return Image.new("RGB", (size, size), (255, 255, 255))


def pick_kontext_resolution(w: int, h: int) -> tuple[int, int]:
    PREFERRED_KONTEXT_RESOLUTIONS = [
        (672, 1568),(688, 1504),(720, 1456),(752, 1392),
        (800, 1328),(832, 1248),(880, 1184),(944, 1104),
        (1024, 1024),(1104, 944),(1184, 880),(1248, 832),
        (1328, 800),(1392, 752),(1456, 720),(1504, 688),(1568, 672),
    ]
    # 计算目标宽高比
    target_ratio = w / h
    return min(
        PREFERRED_KONTEXT_RESOLUTIONS,
        key=lambda wh: abs((wh[0] / wh[1]) - target_ratio)
    )

# @contextmanager
# def region_e(pipeline):
#         yield
#     finally:

class FluxAttnRecorderCallback:
    def __init__(self, record_every=1, name_contains=None, max_modules=None, to_cpu=True):
        self.record_every = max(1, int(record_every))
        self.name_contains = name_contains
        self.max_modules = max_modules
        self.to_cpu = to_cpu
        self.records = []
    def __call__(self, pipe, step_index, timestep, callback_kwargs):

        h = callback_kwargs.get("height", None) // 16
        w = callback_kwargs.get("width", None) // 16
        all_maps_a = []
        all_maps_b = []
        all_maps_c = []
        all_maps_d = []
        all_maps_e = []
        all_maps_f = []
        found_module_names = [] # 记录符合条件的模块名
        for name, module in pipe.transformer.named_modules():
            if isinstance(module, FluxAttention) or isinstance(module,FluxAttentionZero):
 
                attn_maps_a = getattr(module, "attention_probs_query_a_key_noise", None)
                attn_maps_b = getattr(module, "attention_probs_query_b_key_noise", None)
                attn_maps_c = getattr(module, "attention_probs_query_c_key_noise", None)
                attn_maps_d = getattr(module, "attention_probs_query_d_key_noise", None)
                attn_maps_e = getattr(module, "attention_probs_query_e_key_noise", None)
                attn_maps_f = getattr(module, "attention_probs_query_f_key_noise", None)
                if attn_maps_a is not None and attn_maps_b is not None:
                    _pred_i_a = attn_maps_a[0][1][0]
                    _pred_i_b = attn_maps_b[0][1][0]
                    _pred_i_c = attn_maps_c[0][1][0]
                    _pred_i_d = attn_maps_d[0][1][0]
                    _pred_i_e = attn_maps_e[0][1][0]
                    _pred_i_f = attn_maps_f[0][1][0]
                    def minmax_norm(x, eps=1e-8):
                        return (x - x.min()) / (x.max() - x.min() + eps)
                    
                    # h, w = latents.shape[-2]//2, latents.shape[-1]//2
                    norm_a = minmax_norm(_pred_i_a).view(h, w).to(torch.float32).cpu().numpy()
                    norm_b = minmax_norm(_pred_i_b).view(h, w).to(torch.float32).cpu().numpy()
                    norm_d = minmax_norm(_pred_i_d).view(h, w).to(torch.float32).cpu().numpy()
                    norm_e = minmax_norm(_pred_i_e).view(h, w).to(torch.float32).cpu().numpy()
                    norm_f = minmax_norm(_pred_i_f).view(h, w).to(torch.float32).cpu().numpy()
                    
                    all_maps_a.append(norm_a)
                    all_maps_b.append(norm_b)
                    all_maps_d.append(norm_d)
                    all_maps_e.append(norm_e)
                    all_maps_f.append(norm_f)
                    found_module_names.append(name) # 存储名字
        
        if found_module_names:
            self.records.append({
                "step_index": int(step_index),
                "timestep": int(timestep.item()) if torch.is_tensor(timestep) else int(timestep),
                "module_names": found_module_names,
                "pred_a": all_maps_a,
                "pred_b": all_maps_b,
                # "pred_c": all_maps_c,
                "pred_d": all_maps_d,
                "pred_e": all_maps_e,
                "pred_f": all_maps_f,
            })

        return callback_kwargs



class FluxAttnRecorderCallbackPEFT:
    def __init__(self, record_every=1, name_contains=None, max_modules=None, to_cpu=True):
        self.record_every = max(1, int(record_every))
        self.name_contains = name_contains
        self.max_modules = max_modules
        self.to_cpu = to_cpu
        self.records = []
    def __call__(self, pipe, step_index, timestep, callback_kwargs):
        h = callback_kwargs.get("height", None) // 16
        w = callback_kwargs.get("width", None) // 16
        all_maps_a = []
        all_maps_b = []
        all_maps_c = []
        all_maps_d = []
        all_maps_e = []
        all_maps_f = []
        found_module_names = [] # 记录符合条件的模块名

        for name, module in pipe.transformer.named_modules():
            if isinstance(module, FluxAttention) or isinstance(module,FluxAttentionZero):
                attn_maps_a = getattr(module, "attention_probs_query_a_key_noise", None)
                attn_maps_b = getattr(module, "attention_probs_query_b_key_noise", None)
                attn_maps_c = getattr(module, "attention_probs_query_c_key_noise", None)
                attn_maps_d = getattr(module, "attention_probs_query_d_key_noise", None)
                attn_maps_e = getattr(module, "attention_probs_query_e_key_noise", None)
                attn_maps_f = getattr(module, "attention_probs_query_f_key_noise", None)
                if attn_maps_a is not None and attn_maps_b is not None:
                    _pred_i_a = attn_maps_a[0][1][0]
                    _pred_i_b = attn_maps_b[0][1][0]
                    _pred_i_c = attn_maps_c[0][1][0]
                    _pred_i_d = attn_maps_d[0][1][0]
                    _pred_i_e = attn_maps_e[0][1][0]
                    _pred_i_f = attn_maps_f[0][1][0]
                    
                    def minmax_norm(x, eps=1e-8):
                        return (x - x.min()) / (x.max() - x.min() + eps)
                    
                    # h, w = latents.shape[-2]//2, latents.shape[-1]//2
                    norm_a = minmax_norm(_pred_i_a).view(h, w).to(torch.float32).cpu().numpy()
                    norm_b = minmax_norm(_pred_i_b).view(h, w).to(torch.float32).cpu().numpy()
                    norm_d = minmax_norm(_pred_i_d).view(h, w).to(torch.float32).cpu().numpy()
                    norm_e = minmax_norm(_pred_i_e).view(h, w).to(torch.float32).cpu().numpy()
                    norm_f = minmax_norm(_pred_i_f).view(h, w).to(torch.float32).cpu().numpy()
                    
                    all_maps_a.append(norm_a)
                    all_maps_b.append(norm_b)
                    all_maps_d.append(norm_d)
                    all_maps_e.append(norm_e)
                    all_maps_f.append(norm_f)
                    found_module_names.append(name) # 存储名字
        
        if found_module_names:
            self.records.append({
                "step_index": int(step_index),
                "timestep": int(timestep.item()) if torch.is_tensor(timestep) else int(timestep),
                "module_names": found_module_names,
                "pred_a": all_maps_a,
                "pred_b": all_maps_b,
                # "pred_c": all_maps_c,
                "pred_d": all_maps_d,
                "pred_e": all_maps_e,
                "pred_f": all_maps_f,
            })

        return callback_kwargs
