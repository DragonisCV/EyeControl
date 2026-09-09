from viescore import VIEScore
import json
import os
# import megfile
from PIL import Image
from tqdm import tqdm
import argparse
import csv
import glob

def resize_long_side(pil_img, target_long=1024):
    w, h = pil_img.size
    long_side = max(w, h)

    if long_side <= target_long:
        return pil_img  # 不放大

    scale = target_long / long_side
    new_w = int(w * scale)
    new_h = int(h * scale)

    return pil_img.resize((new_w, new_h), Image.BICUBIC)
    
def process_single_item(item, vie_score):
    src_image_path = item["source_path"]
    tgt_image_path = item["save_path"]
    instruction = item["instruction"]

    pil_image_raw = Image.open(src_image_path).convert("RGB")
    pil_image_edited = Image.open(tgt_image_path).convert("RGB")

    # 先对齐尺寸
    if pil_image_raw.size != pil_image_edited.size:
        pil_image_edited = pil_image_edited.resize(pil_image_raw.size)

    # 再统一缩放到 long side 1024
    pil_image_raw = resize_long_side(pil_image_raw, 1024)
    pil_image_edited = resize_long_side(pil_image_edited, 1024)

    # -------- Global evaluation --------
    scores = vie_score.evaluate(
        pil_image_raw,
        pil_image_edited,
        instruction
    )
    fa_score = scores["VF"]
    pq_score = scores["PQ"]
    o_score  = scores["O"]

    return {
        "image_name": os.path.basename(src_image_path),
        "fa_score": fa_score,
        "pq_score": pq_score,
        "o_score": o_score,
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", type=str, required=True)
    parser.add_argument("--target_dir", type=str, required=True)
    parser.add_argument("--prompt_dir", type=str, required=True)
    parser.add_argument("--save_dir", type=str, required=True)
    parser.add_argument("--backbone", type=str, default="google")
    parser.add_argument("--suffix", type=str, default="focus")
    args = parser.parse_args()

    # vie_score = VIEScore(backbone=args.backbone, task="tie", key_path='secret_t2.env')
    vie_score = VIEScore(backbone="qwen25vl", task="tie", key_id=None, vllm_ture=True)

    input_files = sorted(os.listdir(args.input_dir))

    dataset = []

    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
    
    for fname in input_files:
        # 1. 忽略 input_dir 中的非图片文件（如 .DS_Store 等）
        if not fname.lower().endswith(supported_formats):
            continue
            
        base = os.path.splitext(fname)[0]
        
        src_path = os.path.join(args.input_dir, fname)
        
        # 2. 使用 glob 动态查找同名但后缀可能不同的文件
        tgt_matches = glob.glob(os.path.join(args.target_dir, f"{base}.*"))
        
        # 3. 过滤出支持的图片格式，并获取真实路径
        tgt_path = next((f for f in tgt_matches if f.lower().endswith(supported_formats)), None)
        
        prompt_txt = os.path.join(args.prompt_dir, base + ".txt")
        prompt_json = os.path.join(args.prompt_dir, base + ".json")

        # ---- 文件检查 ----
        if not os.path.exists(src_path):
            continue
            
        # 如果找不到匹配的 target 文件
        if not tgt_path or not os.path.exists(tgt_path):
            print(f"Missing target for {base} (Checked formats: {supported_formats})")
            continue
            
        # ---- 读取 prompt ----
        if os.path.exists(prompt_txt):
            instruction = open(prompt_txt).read().strip()
        elif os.path.exists(prompt_json):
            with open(prompt_json, "r") as f:
                data = json.load(f)
                if args.suffix == "global":
                    instruction = data[0]["Global Instruction"]
                elif args.suffix == "local":
                    instruction = data[0]["Local Instruction"]
                elif args.suffix == "focus":
                    instruction = data[0]["Global Instruction"] + " " + data[0]["Local Instruction"]
                elif args.suffix == "mask":
                    instruction = data[0]["Global Instruction"] + " " + data[0]["Local Instruction"] + data[0]["Mask Instruction"]
        else:
            print(f"Missing prompt for {fname}")
            continue

        dataset.append({
            "instruction": instruction,
            "source_path": src_path,
            "save_path": tgt_path,
        })

    print(f"Total valid samples: {len(dataset)}")

    futures = []
    group_list = []
    for item in tqdm(dataset):
        future = process_single_item(item, vie_score)
        futures.append(future)
    
    for future in tqdm(futures, total=len(futures)):
        if future:
            group_list.append(future)

    # Calculate metrics directly
    total_items = len(group_list)

    
    print(f"\n=== Evaluation Results ===")
    print(f"Total items processed: {total_items}")
    
    if total_items > 0:
        # Global metrics (for all items)
        avg_fa_score = sum(item['fa_score'] for item in group_list) / total_items
        avg_pq_score = sum(item['pq_score'] for item in group_list) / total_items
        avg_o_score  = sum(item['o_score']  for item in group_list) / total_items
        log_path = os.path.join(args.save_dir, "evaluation.log")

        with open(log_path, "w") as f:
            f.write("=== Evaluation Results ===\n")
            f.write(f"Total items processed: {total_items}\n\n")
            f.write("Global Metrics:\n")
            f.write(f"Average fa_score: {avg_fa_score:.6f}\n")
            f.write(f"Average pq_score: {avg_pq_score:.6f}\n")
            f.write(f"Average o_score: {avg_o_score:.6f}\n")

            metric_names = ["fa_score", "pq_score", "o_score"]

            for metric in metric_names:
                csv_path = os.path.join(args.save_dir, f"{metric}.csv")
                with open(csv_path, "w", newline="") as csvfile:
                    writer = csv.writer(csvfile)
                    writer.writerow(["image_name", metric])

                    for item in group_list:
                        writer.writerow([item["image_name"], item[metric]])
                        
        print(f"\nGlobal Metrics (all {total_items} items):")
        print(f"  Average fa_score: {avg_fa_score:.4f}")
        print(f"  Average pq_score: {avg_pq_score:.4f}")
        print(f"  Average o_score: {avg_o_score:.4f}")
