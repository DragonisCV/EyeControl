import argparse
import glob
import os
from pyiqa import create_metric
from tqdm import tqdm
import csv
from time import time
import logging
import datetime
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torchvision.transforms as transforms

class ImageDataset(Dataset):
    def __init__(self, input_paths, ref_paths=None, metric_mode='NR'):
        self.input_paths = input_paths
        self.ref_paths = ref_paths
        self.metric_mode = metric_mode
        self.transform = transforms.Compose([
            transforms.ToTensor()
        ])

    def __len__(self):
        return len(self.input_paths)

    def __getitem__(self, idx):
        img_path = self.input_paths[idx]
        img = Image.open(img_path).convert('RGB')

        img = self.transform(img)
        c,h,w = img.shape
        # img = img[:,:,int(w//3*2):]
        if self.metric_mode == 'FR' and self.ref_paths is not None:
            ref_img_path = self.ref_paths[idx]
            ref_img = Image.open(ref_img_path).convert('RGB')
            ref_img = self.transform(ref_img)
            return img, ref_img, img_path
        else:
            return img, img_path

def main(args):


    metric_name = args.metric_name.lower()

    # Define supported image formats
    supported_formats = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')

    if os.path.isfile(args.target):
        input_paths = [args.target] if args.target.lower().endswith(supported_formats) else []
        if args.ref is not None:
            ref_paths = [args.ref] if args.ref.lower().endswith(supported_formats) else []
    else:
        input_paths = [path for path in sorted(glob.glob(os.path.join(args.target, '*'))) if path.lower().endswith(supported_formats)]
        if args.ref is not None:
            ref_paths = [path for path in sorted(glob.glob(os.path.join(args.ref, '*'))) if path.lower().endswith(supported_formats)]

    processed_images = set()
    start_index = 0
    if args.save_file and os.path.exists(args.save_file):
        with open(args.save_file, 'r') as f:
            reader = csv.reader(f)
            for row in reader:
                processed_images.add(row[0])
        start_index = len(processed_images)

    # Check if all images have been processed
    all_images = set(os.path.basename(path) for path in input_paths)
    # import ipdb; ipdb.set_trace()
    if processed_images == all_images:
        log_message = f"All images have been processed. Results are in {args.save_file}."
        print(log_message)
        logging.info(f"{datetime.datetime.now()} - {log_message}")
        return

    # Set up IQA model
    if metric_name != 'fid':
        if metric_name in ['psnr', 'ssim']:
            iqa_model = create_metric(metric_name, test_y_channel=True, color_space='ycbcr')
        elif metric_name in ['nima']:
            iqa_model = create_metric(metric_name, metric_mode=args.metric_mode,backbone_path="pretrained/inception_resnet_v2/inception_resnet_v2/model.safetensors")
        else:
            iqa_model = create_metric(metric_name, metric_mode=args.metric_mode)
        metric_mode = iqa_model.metric_mode

        # Create dataset and dataloader
        dataset = ImageDataset(input_paths[start_index:], ref_paths[start_index:] if args.ref is not None else None, metric_mode)
        batch_size = 1 if metric_name == 'qalign' else args.batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    if args.save_file and metric_name != 'fid':
        sf = open(args.save_file, 'a' if processed_images else 'w')
        sfwriter = csv.writer(sf)

    avg_score = 0
    test_img_num = len(input_paths)
    processed_img_num = 0
    
    if metric_name != 'fid':
        pbar = tqdm(total=test_img_num - start_index, unit='image', initial=start_index)
        for batch in dataloader:
            if metric_mode == 'FR':
                imgs, ref_imgs, img_paths = batch
                scores = iqa_model(imgs, ref_imgs)
            else:
                imgs, img_paths = batch
                scores = iqa_model(imgs)
            if isinstance(scores,torch.Tensor):
                scores = (scores,)
            for score, img_path in zip(scores, img_paths):
                img_name = os.path.basename(img_path)
                if img_name in processed_images:
                    continue

                score = round(score.item(), 4)  # Keep 4 decimal places
                avg_score += score
                processed_img_num += 1
                pbar.update(1)
                # 更新进度条描述而不是追加
                pbar.set_description(f'{metric_name} of {img_name}: {score:.4f}', refresh=True)
                if args.save_file:
                    sfwriter.writerow([img_name, f'{score:.4f}'])

        pbar.close()
        if processed_img_num > 0:
            avg_score /= processed_img_num
    else:
        fid_file = os.path.join(os.path.dirname(args.save_file), 'fid.txt')
        if os.path.exists(fid_file):
            with open(fid_file, 'r') as f:
                content = f.read().strip()
                if content:
                    avg_score = float(content)
                    log_message = f"FID score already exists: {avg_score:.4f}"
                    print(log_message)
                    logging.info(f"{datetime.datetime.now()} - {log_message}")
                    return
        else:
            assert os.path.isdir(args.target), 'For FID, input path must be a folder.'
            iqa_model = create_metric(metric_name, metric_mode=args.metric_mode)
            avg_score = iqa_model(args.target, args.ref)
    
    # if torch.cuda.is_available():
    #     cuda_memory_summary = torch.cuda.memory_summary()
    #     print(cuda_memory_summary)
    #     logging.info(f"{datetime.datetime.now()} - {cuda_memory_summary}")

    msg = f'Average {metric_name} score for {processed_img_num} newly processed images in {args.target}: {avg_score:.4f}'
    print(msg)
    logging.info(f"{datetime.datetime.now()} - {msg}")
    if metric_name == 'fid':
        fid_file = os.path.join(os.path.dirname(args.save_file), 'fid.txt')
        with open(fid_file, 'w') as f:
            f.write(f'{avg_score:.4f}\n')
        log_message = f'Done! FID result saved in fid.txt.'
        print(log_message)
        logging.info(f"{datetime.datetime.now()} - {log_message}")
    elif args.save_file:
        sf.close()
        log_message = f'Done! Results saved in {args.save_file}.'
        print(log_message)
        logging.info(f"{datetime.datetime.now()} - {log_message}")
    else:
        log_message = 'Done!'
        print(log_message)
        logging.info(f"{datetime.datetime.now()} - {log_message}")


if __name__ == '__main__':
    """Inference demo for pyiqa."""
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--target', type=str, default=None, help='Input image/folder path.')
    parser.add_argument('-r', '--ref', type=str, default=None, help='Reference image/folder path (if needed).')
    parser.add_argument('-i', '--input', type=str, default=None, help='Reference image/folder path (if needed).')
    parser.add_argument('--mask', type=str, default=None, help='Reference image/folder path (if needed).')
    parser.add_argument(
        '--metric_mode',
        type=str,
        default='FR',
        help='Metric mode: Full-Reference or No-Reference. Options: FR|NR.')
    parser.add_argument('-m', '--metric_name', type=str, default='PSNR', help='IQA metric name, case sensitive.')
    parser.add_argument('--save_file', type=str, default=None, help='Path to save results.')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size.')

    args = parser.parse_args()
    save_dir = '/'.join(args.save_file.split('/')[:-1])
    os.makedirs(save_dir, exist_ok=True)
    logging.basicConfig(filename=f'{save_dir}/inference_iqa.log', level=logging.INFO)
    main(args)