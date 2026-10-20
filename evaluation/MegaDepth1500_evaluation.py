import os
import sys
import cv2
from pathlib import Path
import numpy as np
import torch
import torch.utils.data as data
import tqdm
from copy import deepcopy
from torchvision.transforms import ToTensor
import torch.nn.functional as F
import json

import scipy.io as scio
import poselib

import json
import argparse
import datetime

parser=argparse.ArgumentParser(description='MegaDepth dataset evaluation script')
parser.add_argument('--name',type=str,default='LiftFeat',help='experiment name')
parser.add_argument('--gpu',type=str,default='0',help='GPU ID')
parser.add_argument('--num_workers',type=int,default=4,help='number of dataloader worker processes for prefetching')
args=parser.parse_args()

os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

sys.path.append(os.path.join(os.path.dirname(__file__),'../'))
from models.liftfeat_wrapper import LiftFeat
from evaluation.eval_utils import *

from torch.utils.data import Dataset,DataLoader

use_cuda = torch.cuda.is_available()
device = "cuda" if use_cuda else "cpu"

DATASET_ROOT = os.path.join(os.path.dirname(__file__),'../data/megadepth_test_1500')
DATASET_JSON = os.path.join(os.path.dirname(__file__),'../data/megadepth_1500.json')

def single_item_collate(batch):
    return batch[0]


class UniqueImagesDataset(Dataset):
    """
    Dataset to pre-load each unique image in MegaDepth-1500 exactly once.
    """
    def __init__(self, json_file, root_dir):
        with open(json_file, 'r') as f:
            data = json.load(f)

        unique = {}
        for d in data:
            unique[d['pair_names'][0]] = d['size0_hw']
            unique[d['pair_names'][1]] = d['size1_hw']

        self.root_dir = root_dir
        self.items = list(unique.items())

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
            f"Dataset {self.root_dir} does not exist! \n \
              > If you didn't download the dataset, use the downloader tool: python3 -m modules.dataset.download -h")

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        rel_path, (h, w) = self.items[idx]
        img = cv2.resize(cv2.imread(f"{self.root_dir}/{rel_path}"), (w, h))
        return rel_path, img


class MegaDepth1500(Dataset):
    """
        Streamlined MegaDepth-1500 dataloader. The camera poses & metadata are stored in a formatted json for facilitating 
        the download of the dataset and to keep the setup as simple as possible.
    """
    def __init__(self, json_file, root_dir, load_images=True):
        # Load the info & calibration from the JSON
        with open(json_file, 'r') as f:
            self.data = json.load(f)

        self.root_dir = root_dir
        self.load_images = load_images

        if not os.path.exists(self.root_dir):
            raise RuntimeError(
            f"Dataset {self.root_dir} does not exist! \n \
              > If you didn't download the dataset, use the downloader tool: python3 -m modules.dataset.download -h")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        data = deepcopy(self.data[idx])

        if self.load_images:
            h1, w1 = data['size0_hw']
            h2, w2 = data['size1_hw']

            image0 = cv2.resize(cv2.imread(f"{self.root_dir}/{data['pair_names'][0]}"),(w1, h1))
            image1 = cv2.resize(cv2.imread(f"{self.root_dir}/{data['pair_names'][1]}"),(w2, h2))

            data['image0'] = torch.tensor(image0.astype(np.float32)/255).permute(2,0,1)
            data['image1'] = torch.tensor(image1.astype(np.float32)/255).permute(2,0,1)
        else:
            data.pop('image0', None)
            data.pop('image1', None)

        for k,v in data.items():
            if k not in ('dataset_name', 'scene_id', 'pair_id', 'pair_names', 'size0_hw', 'size1_hw', 'image0', 'image1'):
                data[k] = torch.tensor(np.array(v, dtype=np.float32))

        return data

if __name__ == "__main__":
    weights_dir = os.path.join(os.path.dirname(__file__), 'weights/gfl3')
    all_weight_paths = list(Path(weights_dir).rglob('*.pth'))
    all_results_summary = {}

    # Phase 1 loader: stream unique images using worker prefetching
    unique_dataset = UniqueImagesDataset(json_file=DATASET_JSON, root_dir=DATASET_ROOT)
    unique_loader_kwargs = {
        'batch_size': 1,
        'shuffle': False,
        'collate_fn': single_item_collate,
    }
    if args.num_workers > 0:
        unique_loader_kwargs.update({
            'num_workers': args.num_workers,
            'pin_memory': False,
            'prefetch_factor': 2,
            'persistent_workers': True
        })
    unique_loader = DataLoader(unique_dataset, **unique_loader_kwargs)

    # Phase 2 loader: metadata-only pair evaluation (instant, zero image I/O)
    pair_dataset = MegaDepth1500(json_file=DATASET_JSON, root_dir=DATASET_ROOT, load_images=False)
    pair_loader = DataLoader(pair_dataset, batch_size=1, shuffle=False)

    CURRENT_DIR = os.getcwd()
    summary_file = os.path.join(CURRENT_DIR, 'all_weights_summary.json')
    evaluated_count = 0

    for weight_path in all_weight_paths:
        weight_path_str = str(weight_path)
        print(f"\n--- Evaluating {weight_path.name} ---")
        
        liftfeat = LiftFeat(weight=weight_path_str)

        metrics = {}
        R_errs = []
        t_errs = []
        inliers = []
        results = []

        cur_time = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        # Phase 1: Pre-extract all unique images into cache
        feature_cache = {}
        for rel_path, img in tqdm.tqdm(unique_loader, desc=f"caching {weight_path.name}"):
            feature_cache[rel_path] = liftfeat.extract(img)

        # Phase 2: In-memory evaluation of all 1500 pairs
        for d in tqdm.tqdm(pair_loader, desc=f"evaluating {weight_path.name}"):
            name0 = d['pair_names'][0][0]
            name1 = d['pair_names'][1][0]

            data0 = feature_cache[name0]
            data1 = feature_cache[name1]

            mkpts0, mkpts1 = liftfeat.match_features(data0, data1)
            error_infos = compute_pose_error_from_matches(mkpts0, mkpts1, d)
            results.append(error_infos)

        del feature_cache
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        print(f'\n==={cur_time}==={args.name} ({weight_path.name})===')
        d_err_auc, errors=compute_maa(results)

        for s_k,s_v in d_err_auc.items():
            print(f'{s_k}: {s_v*100}')
            
        all_results_summary[weight_path.name] = {k: v * 100 for k, v in d_err_auc.items()}
        evaluated_count += 1

        # Overwrite the single summary file every 40 checkpoints (crash-safe atomic write)
        if evaluated_count % 40 == 0:
            tmp_file = summary_file + '.tmp'
            with open(tmp_file, 'w') as f:
                json.dump(all_results_summary, f, indent=4)
            os.replace(tmp_file, summary_file)
            print(f"\n[Progress Saved] Safely updated {summary_file} ({evaluated_count} checkpoints).")

    # Final save to ensure all results are written to the single file
    tmp_file = summary_file + '.tmp'
    with open(tmp_file, 'w') as f:
        json.dump(all_results_summary, f, indent=4)
    os.replace(tmp_file, summary_file)
        
    print(f"\nEvaluation complete! Final summary saved to {summary_file} (total: {evaluated_count} checkpoints).")