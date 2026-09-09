import os
import cv2
import numpy as np

from torch.utils.data import Dataset
from einops import rearrange
import random
from torchvision import transforms
from torchvision.transforms import Resize,ToTensor
from torchvision.transforms.functional import crop
import torch
from PIL import Image
import json
from src.data.mask_generator import OnlineMaskAugmentor
from diffusers.training_utils import (
    find_nearest_bucket,
)
from tqdm import tqdm


class RandomColorAdjustments():
    def __init__(self,config):
        """
            config: 一个字典，内包含所有的range
        """
        self.config = config

    def get_random_adjustments(self):
        random_adjustments = {}
        for adjust_name,adjust_range in self.config.items():
            random_adjustments[adjust_name] = random.uniform(adjust_range[0],adjust_range[1])

        return random_adjustments

    def __call__(self,img,mask=None):
        random_adjustment = self.get_random_adjustments()
        adjusted_img = apply_basic_adjustments(img,**random_adjustment)
        if mask is not None:
            mask = mask[:,:,np.newaxis]
            adjusted_img = adjusted_img * mask + img * (1 - mask)
        return adjusted_img,random_adjustment


# Example: Dataset integration
class FocusDataset(Dataset):
    """
    Minimal example dataset:
      - loads images (optional) and intent.npz
      - generates online-augmented mask each __getitem__
    """

    def __init__(self, opt):
        """
        items: list of dicts like:
          {"image_path": "...", "intent_path": "..."}  # image_path optional
        """
        self.opt = opt
        self.augmentor_cfg = opt['augmentor_cfg']
        self.polarity = opt['polarity']
        self.buckets = opt['buckets']
        self.caption_mode = opt.get('caption_mode', '')

        self.json_path = opt['json_path']
        input_paths = []
        gt_paths = []
        intent_paths = []
        captions = []
        with open(self.json_path,'r') as f:
            data = json.load(f)
        
        for sample in data:
            image_from = sample['image_from']
            image_to = sample['image_to']
            intent_path = sample['intent_path']
            input_paths.append(image_from)
            gt_paths.append(image_to)
            intent_paths.append(intent_path)

            global_caption = sample['Global Instruction']
            local_caption = sample['Local Instruction']
            captions.append({
                'global_caption':global_caption,
                'local_caption':local_caption
            })
        self.global_caption = "Retouch with natural, professional global color and tone."
        self.local_caption = "Enhance mask part of image as the focal point, with subtle global support."
            
        self.caption = self.global_caption + " " + self.local_caption
        samples = []
        for i in tqdm(range(len(intent_paths)),desc="Sort Bucketing.."):
            ip = input_paths[i]
            gp = gt_paths[i]
            itp = intent_paths[i]
            # 获取尺寸用于分桶（尽量快）
            try:
                with Image.open(ip) as im:
                    w, h = im.size
            except Exception:
                # 兜底：如果 input 图坏了，就用 gt 或者默认
                try:
                    with Image.open(gp) as im:
                        w, h = im.size
                except Exception:
                    # 彻底坏：给一个最常见的尺寸兜底
                    h, w = self.buckets[0]

            bucket_idx = find_nearest_bucket(h, w, self.buckets)
            sample = {
                "input_path": ip,
                "gt_path": gp,
                "intent_path": itp,
                "bucket_idx": bucket_idx,
            }
            if 'gc' in self.caption_mode:
                sample['global_caption'] = captions[i]['global_caption']
                sample['local_caption'] = self.local_caption
                sample['caption'] = sample['global_caption'] + " " + sample['local_caption']
            elif 'glc' in self.caption_mode:
                sample['global_caption'] = captions[i]['global_caption']
                sample['local_caption'] = captions[i]['local_caption']
                sample['caption'] = sample['global_caption'] + " " + sample['local_caption']
            else:
                sample['global_caption'] = self.global_caption
                sample['local_caption'] = self.local_caption
                sample['caption'] = self.global_caption + " " + self.local_caption             
            samples.append(sample)
        self.samples = samples
        
        self.role="default"

        self.preprocess_by_bucket = {}
        for bi, (bh, bw) in enumerate(self.buckets):
            ops = [transforms.Resize((bh, bw))]
            self.preprocess_by_bucket[bi] = transforms.Compose(ops + [transforms.ToTensor()])


        # ])

        self.aug = OnlineMaskAugmentor(self.augmentor_cfg, seed=self.augmentor_cfg.get("seed", None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        input_path = s["input_path"]
        gt_path = s["gt_path"]
        intent_path = s["intent_path"]
        bucket_idx = s["bucket_idx"]

        global_caption = s['global_caption']
        local_caption = s['local_caption']
        caption = s['caption']
        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        # build intent dict for augmentor (np.load returns NpzFile)
        intent = np.load(intent_path, allow_pickle=False)
        intent_dict = {k: intent[k] for k in intent.files}

        out = self.aug(intent_dict, polarity=self.polarity)
        coarse_region_mask = intent_dict[f'coarse_region_{self.polarity}']
        mask_gt = Image.fromarray((coarse_region_mask*255.0).clip(0,255).astype(np.uint8))
        mask = out["mask"]  # torch.FloatTensor [H,W]
        mask = mask.numpy()
        mask = np.stack([mask,mask,mask],axis=-1)
        mask = Image.fromarray((mask*255.0).clip(0,255).astype(np.uint8))

        preprocess = self.preprocess_by_bucket[bucket_idx]

        input_img = preprocess(input_img)
        gt_img = preprocess(gt_img)
        mask = preprocess(mask)
        mask_gt = preprocess(mask_gt)

        sample = {
            'img': input_img,
            'gt_img': gt_img,
            'mask':mask,
            'mask_gt': mask_gt,
            'mask_path':intent_path,
            'img_path': input_path,
            'global_caption': global_caption,
            'local_caption': local_caption,
            'caption': caption,
            'role': self.role,
        }

        #     "mask": mask,
        #     "tool": out["tool"],
        # }
        #     sample["debug"] = out["debug"]
        return sample



class IntentDecoupleDataset(Dataset):
    """
    Minimal example dataset:
      - loads images (optional) and intent.npz
      - generates online-augmented mask each __getitem__
    """

    def __init__(self, opt):
        """
        items: list of dicts like:
          {"image_path": "...", "intent_path": "..."}  # image_path optional
        """
        self.opt = opt
        self.augmentor_cfg = opt['augmentor_cfg']
        self.polarity = opt['polarity']
        self.intent_root = opt['intent_root']
        self.img_root = opt['img_root']
        self.buckets = opt['buckets']
        self.caption_mode = opt.get('caption_mode', '')
        input_paths = []
        gt_paths = []
        intent_paths = sorted([os.path.join(self.intent_root,f) for f in os.listdir(self.intent_root)])

        captions = []
        if 'gc' in self.caption_mode:
            caption_dict = {}
            with open(opt['caption_path'],'r') as f:
                caption_data = json.load(f)
            for sample in tqdm(caption_data,desc="Building Caption Dict..."):
                global_caption = sample['Global Instruction']
                local_caption = sample['Local Instruction']
                image_from = sample['image_from']
                image_to = sample['image_to']
                id = os.path.basename(os.path.dirname(image_from))
                fid = os.path.basename(image_from).split('.')[0].split('_')[-1]
                tid = os.path.basename(image_to).split('.')[0].split('_')[-1]
                print(f'{id}_{fid}_{tid}')
                caption_dict[f'{id}_{fid}_{tid}'] = {
                    "global_caption": global_caption,
                    "local_caption": local_caption
                }
            self.caption_dict = caption_dict


        for intent_path in intent_paths:
            split_path = os.path.basename(intent_path).split(".")[0].split('_')
            id,input_id,gt_id = split_path[0],split_path[-3],split_path[-1]
            input_paths.append(os.path.join(self.img_root,id,f"frame_{input_id}.png"))
            gt_paths.append(os.path.join(self.img_root,id,f"frame_{gt_id}.png"))
            if 'gc' in self.caption_mode:
                captions.append(self.caption_dict[f'{id}_{input_id}_{gt_id}'])


        self.global_caption = "Retouch with natural, professional global color and tone."
        self.local_caption = "Enhance mask part of image as the focal point, with subtle global support."
            

        self.caption = self.global_caption + " " + self.local_caption
        samples = []
        for i in tqdm(range(len(intent_paths)),desc="Sort Bucketing.."):
            ip = input_paths[i]
            gp = gt_paths[i]
            itp = intent_paths[i]

            # 获取尺寸用于分桶（尽量快）
            try:
                with Image.open(ip) as im:
                    w, h = im.size
            except Exception:
                # 兜底：如果 input 图坏了，就用 gt 或者默认
                try:
                    with Image.open(gp) as im:
                        w, h = im.size
                except Exception:
                    # 彻底坏：给一个最常见的尺寸兜底
                    h, w = self.buckets[0]

            bucket_idx = find_nearest_bucket(h, w, self.buckets)
            sample = {
                "input_path": ip,
                "gt_path": gp,
                "intent_path": itp,
                "bucket_idx": bucket_idx,
            }
            if 'gc' in self.caption_mode:
                sample['global_caption'] = captions['global_caption']
                sample['local_caption'] = self.local_caption
                sample['caption'] = sample['global_caption'] + " " + sample['local_caption']
            else:
                sample['global_caption'] = self.global_caption
                sample['local_caption'] = self.local_caption
                sample['caption'] = self.global_caption + " " + self.local_caption             
            samples.append(sample)
        self.samples = samples
        
        

        self.role="default"

        self.preprocess_by_bucket = {}
        for bi, (bh, bw) in enumerate(self.buckets):
            ops = [transforms.Resize((bh, bw))]
            self.preprocess_by_bucket[bi] = transforms.Compose(ops + [transforms.ToTensor()])


        # ])

        self.aug = OnlineMaskAugmentor(self.augmentor_cfg, seed=self.augmentor_cfg.get("seed", None))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        input_path = s["input_path"]
        gt_path = s["gt_path"]
        intent_path = s["intent_path"]
        bucket_idx = s["bucket_idx"]

        global_caption = s['global_caption']
        local_caption = s['local_caption']
        caption = s['caption']
        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        # build intent dict for augmentor (np.load returns NpzFile)
        intent = np.load(intent_path, allow_pickle=False) 
        intent_dict = {k: intent[k] for k in intent.files}

        out = self.aug(intent_dict, polarity=self.polarity)
        mask = out["mask"]  # torch.FloatTensor [H,W]
        mask = mask.numpy()
        mask = np.stack([mask,mask,mask],axis=-1)
        mask = Image.fromarray((mask*255.0).clip(0,255).astype(np.uint8))

        preprocess = self.preprocess_by_bucket[bucket_idx]

        input_img = preprocess(input_img)
        gt_img = preprocess(gt_img)
        mask = preprocess(mask)

        sample = {
            'img': input_img,
            'gt_img': gt_img,
            'mask':mask,
            'mask_path':intent_path,
            'img_path': input_path,
            'global_caption': global_caption,
            'local_caption': local_caption,
            'caption': caption,
            'role': self.role,
        }

        #     "mask": mask,
        #     "tool": out["tool"],
        # }
        #     sample["debug"] = out["debug"]
        return sample


class GlobalDataset(Dataset):
    """
    Minimal example dataset:
      - loads images (optional) and intent.npz
      - generates online-augmented mask each __getitem__
    """

    def __init__(self, opt):
        """
        items: list of dicts like:
          {"image_path": "...", "intent_path": "..."}  # image_path optional
        """
        self.opt = opt
        self.json_paths = opt['json_paths']
        with open(self.json_paths,'r') as f:
            data = json.load(f)
        self.buckets = opt['buckets']


        input_paths = []
        target_paths = []
        prompts =[]
        for datum in data:
            input_paths.append(datum['image_from'])
            target_paths.append(datum['image_to'])
            prompts.append(datum['Global Instruction'])
        

        samples = []
        for i in tqdm(range(len(input_paths)),desc="Sort Bucketing.."):
            ip = input_paths[i]
            gp = target_paths[i]
            prompt = prompts[i]

            # 获取尺寸用于分桶（尽量快）
            try:
                with Image.open(ip) as im:
                    w, h = im.size
            except Exception:
                # 兜底：如果 input 图坏了，就用 gt 或者默认
                try:
                    with Image.open(gp) as im:
                        w, h = im.size
                except Exception:
                    # 彻底坏：给一个最常见的尺寸兜底
                    h, w = self.buckets[0]

            bucket_idx = find_nearest_bucket(h, w, self.buckets)
            samples.append(
                {
                    "input_path": ip,
                    "gt_path": gp,
                    "prompt": prompt,
                    "bucket_idx": bucket_idx,
                    # "cot":cot,
                    # "role":role,
                }
            )
        self.samples = samples

        self.preprocess_by_bucket = {}
        for bi, (bh, bw) in enumerate(self.buckets):
            ops = [transforms.Resize((bh, bw))]
            self.preprocess_by_bucket[bi] = transforms.Compose(ops + [transforms.ToTensor()])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        input_path = s["input_path"]
        gt_path = s["gt_path"]
        bucket_idx = s["bucket_idx"]
        prompt = s['prompt']

        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')


        preprocess = self.preprocess_by_bucket[bucket_idx]

        input_img = preprocess(input_img)
        gt_img = preprocess(gt_img)
        mask = torch.zeros_like(gt_img)

        sample = {
            'img': input_img,
            'gt_img': gt_img,
            'mask':mask,
            'img_path': input_path,
            'global_caption': prompt,
            'caption': prompt,
            # 'role': role,
        }

        #     "mask": mask,
        #     "tool": out["tool"],
        # }
        #     sample["debug"] = out["debug"]
        return sample

class LocalDataset(Dataset):
    """
    Minimal example dataset:
      - loads images (optional) and intent.npz
      - generates online-augmented mask each __getitem__
    """

    def __init__(self, opt):
        """
        items: list of dicts like:
          {"image_path": "...", "intent_path": "..."}  # image_path optional
        """
        self.opt = opt
        self.json_paths = opt['json_paths']
        with open(self.json_paths,'r') as f:
            data = json.load(f)
        self.buckets = opt['buckets']

        self.enlarge_ratio = opt.get('enlarge_ratio',1)
        input_paths = []
        target_paths = []
        intent_paths = []
        prompts =[]
        for datum in data:
            image_from = datum['image_from']
            image_to = datum['image_to']
            intent_path_list = datum['intent_mask_list']
            prompts_list = datum['prompts']
            
            for intent_path in intent_path_list:
                for prompt in prompts_list:
                    input_paths.append(image_from)
                    target_paths.append(image_to)
                    intent_paths.append(intent_path)
                    prompts.append(prompt)

        

        samples = []
        for i in tqdm(range(len(input_paths)),desc="Sort Bucketing.."):
            ip = input_paths[i]
            gp = target_paths[i]
            mp  = intent_paths[i]
            prompt = prompts[i]
            # 获取尺寸用于分桶（尽量快）
            try:
                with Image.open(ip) as im:
                    w, h = im.size
            except Exception:
                # 兜底：如果 input 图坏了，就用 gt 或者默认
                try:
                    with Image.open(gp) as im:
                        w, h = im.size
                except Exception:
                    # 彻底坏：给一个最常见的尺寸兜底
                    h, w = self.buckets[0]

            bucket_idx = find_nearest_bucket(h, w, self.buckets)
            samples.append(
                {
                    "input_path": ip,
                    "gt_path": gp,
                    'intent_path': mp,
                    "prompt": prompt,
                    "bucket_idx": bucket_idx,
                }
            )
        self.samples = samples * self.enlarge_ratio
        print(len(self.samples))
        print(len(self.samples))
        print(len(self.samples))
        print(len(self.samples))
        print(len(self.samples))
        print(len(self.samples))

        self.preprocess_by_bucket = {}
        for bi, (bh, bw) in enumerate(self.buckets):
            ops = [transforms.Resize((bh, bw))]
            self.preprocess_by_bucket[bi] = transforms.Compose(ops + [transforms.ToTensor()])


    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int):
        s = self.samples[idx]
        input_path = s["input_path"]
        gt_path = s["gt_path"]
        intent_path = s['intent_path']
        bucket_idx = s["bucket_idx"]
        prompt = s['prompt']['Local Instruction']

        input_img = Image.open(input_path).convert('RGB')
        gt_img = Image.open(gt_path).convert('RGB')
        mask = Image.open(intent_path).convert('RGB')

        preprocess = self.preprocess_by_bucket[bucket_idx]

        input_img = preprocess(input_img)
        gt_img = preprocess(gt_img)
        mask = preprocess(mask)

        sample = {
            'img': input_img,
            'gt_img': gt_img,
            'mask':mask,
            'mask_path':intent_path,
            'img_path': input_path,
            'global_caption': prompt,
            'local_caption': prompt,
            'caption': prompt,
        }

        #     "mask": mask,
        #     "tool": out["tool"],
        # }
        #     sample["debug"] = out["debug"]
        return sample
