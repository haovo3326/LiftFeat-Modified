"""
	"LiftFeat: 3D Geometry-Aware Local Feature Matching"
    training script
"""

import argparse
import os
import time
import sys
from dataclasses import dataclass
from typing import Optional, Tuple

sys.path.append(os.path.dirname(__file__))

KAGGLE_HOST_NAMES = ["haovo3326", "thanhbih", "makago", "md090306", "dngcharles", "teukun"]

@dataclass
class TrainerConfig:
    name: str
    platform: str
    kaggle_host_id: int
    use_megadepth: bool
    megadepth_root_path: str
    megadepth_batch_size: int
    use_coco: bool
    coco_root_path: str
    coco_batch_size: int
    ckpt_save_path: str
    latest_ckpt_path: Optional[str]
    n_steps: int
    scheduler_steps: int
    lr: float
    gamma_steplr: float
    training_res: Tuple[int, int]
    device_num: str
    dry_run: bool
    save_ckpt_every: int
    use_coord_loss: bool
    num_workers: int
    amp: bool
    log_every: int
    max_load_failures: int


def parse_training_res(value):
    try:
        width, height = map(int, value.split(','))
    except ValueError as ex:
        raise argparse.ArgumentTypeError('Expected width,height, for example 800,608.') from ex
    if width <= 0 or height <= 0:
        raise argparse.ArgumentTypeError('Training resolution values must be positive.')
    return width, height


def parse_arguments():
    parser = argparse.ArgumentParser(description="LiftFeat training script.")

    runtime = parser.add_argument_group('Runtime')
    runtime.add_argument('--name', type=str, default='LiftFeat', help='Run name used for process title and checkpoints.')
    runtime.add_argument('--device_num', type=str, default='0', help='CUDA device id to expose. Default is "0".')
    runtime.add_argument('--dry_run', action='store_true', help='Run one mini-batch as a sanity check.')
    runtime.add_argument('--num_workers', type=int, default=2, help='DataLoader workers per dataset.')
    runtime.add_argument('--log_every', type=int, default=50, help='Log to TensorBoard every N steps.')
    runtime.add_argument('--max_load_failures', type=int, default=50, help='Max consecutive corrupted batches allowed.')

    platform = parser.add_argument_group('Platform')
    platform.add_argument('--platform', type=str, default='Kaggle', choices=['Kaggle', 'Server'],
                          help='Training platform. Kaggle uses split /kaggle/input dataset shards; Server uses one MegaDepth root.')
    platform.add_argument('--kaggle_host_id', type=int, default=0,
                          help='Kaggle host ID used to select a host/user name for /kaggle/input paths.')

    megadepth = parser.add_argument_group('MegaDepth dataset')
    megadepth.add_argument('--use_megadepth', action='store_true')
    megadepth.add_argument('--megadepth_root_path', type=str, default='/kaggle/input/datasets')
    megadepth.add_argument('--megadepth_batch_size', type=int, default=6)

    coco = parser.add_argument_group('COCO20k dataset')
    coco.add_argument('--use_coco', action='store_true')
    coco.add_argument('--coco_root_path', type=str, default='/home/yepeng_liu/code_python/dataset/coco_20k')
    coco.add_argument('--coco_batch_size', type=int, default=4)

    checkpoints = parser.add_argument_group('Checkpoints')
    checkpoints.add_argument('--ckpt_save_path', type=str, default='/kaggle/working/trained_weights/megadepth')
    checkpoints.add_argument('--latest_ckpt_path', type=str, default=None)
    checkpoints.add_argument('--save_ckpt_every', type=int, default=2000)

    optimization = parser.add_argument_group('Optimization')
    optimization.add_argument('--n_steps', type=int, default=160_000, help='Number of training steps.')
    optimization.add_argument('--scheduler_steps', type=int, default=10000, help='Step interval for StepLR.')
    optimization.add_argument('--lr', type=float, default=3e-4, help='Learning rate.')
    optimization.add_argument('--gamma_steplr', type=float, default=0.5, help='Gamma value for StepLR.')
    optimization.add_argument('--training_res', type=parse_training_res, default=(800, 608))
    optimization.add_argument('--amp', action='store_true', help='Enable Automatic Mixed Precision (FP16).')

    losses = parser.add_argument_group('Losses')
    losses.add_argument('--use_coord_loss', action='store_true', help='Enable coordinate loss.')

    args = parser.parse_args()
    
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device_num
    return TrainerConfig(**vars(args))

config = parse_arguments()

import torch
from torch import optim
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import tqdm
import glob

from models.model import LiftFeatSPModel
from loss.loss import LiftFeatLoss
from utils.config import original_fusion_featureboost_config
from utils.depth_anything_wrapper import DepthAnythingExtractor
from utils.alike_wrapper import ALikeExtractor

from dataset import megadepth_wrapper
from dataset import coco_wrapper
from dataset.megadepth import MegaDepthDataset
from dataset.coco_augmentor import COCOAugmentor

import setproctitle


def move_optimizer_state_to_device(optimizer, device):
    for state in optimizer.state.values():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                state[key] = value.to(device)


def resolve_megadepth_paths(config):
    if config.platform == "Kaggle":
        host_name = KAGGLE_HOST_NAMES[config.kaggle_host_id]
        train_base_path = f"{config.megadepth_root_path}/{host_name}/megadepth-metadata/train_data/megadepth_indices"
        trainval_data_source = [
            f"{config.megadepth_root_path}/kashiwaba/megadepth-v1-p1/MegaDepth_v1_p1",
            f"{config.megadepth_root_path}/kashiwaba/megadepth-v1-p2/MegaDepth_v1_p2",
            f"{config.megadepth_root_path}/kashiwaba/megadepth-v1-p3/MegaDepth_v1_p3",
            f"{config.megadepth_root_path}/kashiwaba/megadepth-v1-p4/MegaDepth_v1_p4"
        ]
    else:
        train_base_path = f"{config.megadepth_root_path}/train_data/megadepth_indices"
        trainval_data_source = f"{config.megadepth_root_path}/MegaDepth_v1"

    return trainval_data_source, f"{train_base_path}/scene_info_0.1_0.7"


class Trainer:
    def __init__(self, config):
        self.config = config
        self.steps = config.n_steps
        self.dry_run = config.dry_run
        self.save_ckpt_every = config.save_ckpt_every
        self.ckpt_save_path = config.ckpt_save_path
        self.model_name = config.name
        self.use_coord_loss = config.use_coord_loss
        self.use_coco = config.use_coco
        self.coco_batch_size = config.coco_batch_size if config.use_coco else 0
        self.use_megadepth = config.use_megadepth
        self.megadepth_batch_size = config.megadepth_batch_size if config.use_megadepth else 0
        self.current_step = 0

        self.print_config()
        self.setup_device()
        self.setup_models()
        self.setup_optimizer()
        self.setup_coco()
        self.setup_megadepth()
        self.setup_logging()
        self.load_checkpoint()

    def print_config(self):
        print(f'Platform: {self.config.platform}')
        print(f'MegaDepth: {self.use_megadepth}-{self.megadepth_batch_size}')
        print(f'COCO20k: {self.use_coco}-{self.coco_batch_size}')
        print(f'Coordinate loss: {self.use_coord_loss}')
        print(f'AMP enabled: {self.config.amp}')
        print(f'Workers: {self.config.num_workers}')

    def setup_device(self):
        self.dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'Training device: {self.dev}')
        if torch.cuda.is_available():
            print(f'GPU: {torch.cuda.get_device_name(0)}')
            
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.amp)

    def setup_models(self):
        self.net = LiftFeatSPModel(original_fusion_featureboost_config).to(self.dev)
        self.loss_fn = LiftFeatLoss(self.dev, lam_descs=1, lam_kpts=2, lam_heatmap=1)
        self.depth_net = DepthAnythingExtractor('vits', self.dev, 256)
        self.alike_net = ALikeExtractor('alike-t', self.dev)

    def setup_optimizer(self):
        self.opt = optim.Adam(filter(lambda x: x.requires_grad, self.net.parameters()), lr=self.config.lr)
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.opt,
            step_size=self.config.scheduler_steps,
            gamma=self.config.gamma_steplr
        )

    def setup_coco(self):
        if self.use_coco:
            self.augmentor = COCOAugmentor(
                img_dir=self.config.coco_root_path,
                device=self.dev,
                load_dataset=True,
                batch_size=self.coco_batch_size,
                out_resolution=self.config.training_res,
                warp_resolution=self.config.training_res,
                sides_crop=0.1,
                max_num_imgs=3000,
                num_test_imgs=5,
                photometric=True,
                geometric=True,
                reload_step=4000
            )

    def setup_megadepth(self):
        if self.use_megadepth:
            trainval_data_source, train_npz_root = resolve_megadepth_paths(self.config)
            npz_paths = glob.glob(train_npz_root + '/*.npz')[:]
            if len(npz_paths) == 0:
                raise RuntimeError(f'No MegaDepth index files found in {train_npz_root}')

            megadepth_dataset = torch.utils.data.ConcatDataset([
                MegaDepthDataset(root_dirs=trainval_data_source, npz_path=path)
                for path in tqdm.tqdm(npz_paths, desc="[MegaDepth] Loading metadata")
            ])
            
            # Integrated multiprocessing and pinned memory constraints
            dataloader_kwargs = {
                'batch_size': self.megadepth_batch_size,
                'shuffle': True,
                'drop_last': True
            }
            if self.config.num_workers > 0 and torch.cuda.is_available():
                dataloader_kwargs.update({
                    'num_workers': self.config.num_workers,
                    'pin_memory': True,
                    'persistent_workers': True,
                    'prefetch_factor': 2
                })

            self.megadepth_dataloader = DataLoader(megadepth_dataset, **dataloader_kwargs)
            self.megadepth_data_iter = iter(self.megadepth_dataloader)

    def setup_logging(self):
        os.makedirs(self.ckpt_save_path, exist_ok=True)
        os.makedirs(self.ckpt_save_path + '/logdir', exist_ok=True)
        self.writer = SummaryWriter(
            self.ckpt_save_path + f'/logdir/{self.model_name}_' + time.strftime("%Y_%m_%d-%H_%M_%S")
        )

    def load_checkpoint(self):
        ckpt_path = self.config.latest_ckpt_path
        if ckpt_path is not None and os.path.isfile(ckpt_path):
            print(f'Loading checkpoint: {ckpt_path}')
            checkpoint = torch.load(ckpt_path, map_location='cpu')
            self.net.load_state_dict(checkpoint['model'])
            self.opt.load_state_dict(checkpoint['optimizer'])
            self.scheduler.load_state_dict(checkpoint['scheduler'])
            move_optimizer_state_to_device(self.opt, self.dev)
            self.current_step = checkpoint.get('step', checkpoint.get('current step', 0))
            if self.config.amp and 'scaler' in checkpoint:
                self.scaler.load_state_dict(checkpoint['scaler'])
            print(f'Resuming from step {self.current_step}.')
        else:
            if ckpt_path is None:
                print('No checkpoint path provided. Starting training from scratch.')
            else:
                print(f'Checkpoint not found: {ckpt_path}. Starting training from scratch.')
        print(f'LR: {self.opt.param_groups[0]["lr"]}; StepLR step_size: {self.scheduler.step_size}; gamma: {self.scheduler.gamma}')
        
    def generate_train_data(self):
        imgs1_t,imgs2_t=[],[]
        imgs1_np,imgs2_np=[],[]
        positives_coarse=[]
        
        if self.use_coco:
                coco_imgs1, coco_imgs2, H1, H2 = coco_wrapper.make_batch(self.augmentor, 0.1)
                h_coarse, w_coarse = coco_imgs1[0].shape[-2] // 8, coco_imgs1[0].shape[-1] // 8
                _ , positives_coco_coarse = coco_wrapper.get_corresponding_pts(coco_imgs1, coco_imgs2, H1, H2, self.augmentor, h_coarse, w_coarse)
                coco_imgs1=coco_imgs1.mean(1,keepdim=True);coco_imgs2=coco_imgs2.mean(1,keepdim=True)
                imgs1_t.append(coco_imgs1);imgs2_t.append(coco_imgs2)
                positives_coarse += positives_coco_coarse
                    
        if self.use_megadepth:
            try:
                megadepth_data=next(self.megadepth_data_iter)
            except StopIteration:
                print('End of MD DATASET')
                self.megadepth_data_iter=iter(self.megadepth_dataloader)
                try:
                    megadepth_data=next(self.megadepth_data_iter)
                except Exception as ex:
                    print(f'MegaDepth data loading failed: {ex}')
                    return None
            except Exception as ex:
                print(f'MegaDepth data loading failed: {ex}')
                return None

            if megadepth_data is None:
                return None

            for k in megadepth_data.keys():
                if isinstance(megadepth_data[k], torch.Tensor):
                    megadepth_data[k] = megadepth_data[k].to(self.dev, non_blocking=True)
            
            megadepth_imgs1_t, megadepth_imgs2_t = megadepth_data['image0'], megadepth_data['image1']
            megadepth_imgs1_t = megadepth_imgs1_t.mean(1, keepdim=True)
            megadepth_imgs2_t = megadepth_imgs2_t.mean(1, keepdim=True)
            imgs1_t.append(megadepth_imgs1_t)
            imgs2_t.append(megadepth_imgs2_t)
            megadepth_imgs1_np, megadepth_imgs2_np = megadepth_data['image0_np'], megadepth_data['image1_np']
            for np_idx in range(megadepth_imgs1_np.shape[0]):
                img1_np, img2_np = megadepth_imgs1_np[np_idx].squeeze(0).cpu().numpy(), megadepth_imgs2_np[
                    np_idx].squeeze(0).cpu().numpy()
                imgs1_np.append(img1_np)
                imgs2_np.append(img2_np)
            positives_megadepth_coarse = megadepth_wrapper.spvs_coarse(megadepth_data, 8)
            positives_coarse += positives_megadepth_coarse
                
        with torch.no_grad():
            if len(imgs1_t) == 0 or len(imgs2_t) == 0:
                raise RuntimeError('No training images were generated. Check that --use_megadepth or --use_coco is enabled.')
            imgs1_t=torch.cat(imgs1_t,dim=0)
            imgs2_t=torch.cat(imgs2_t,dim=0)
            
        return imgs1_t,imgs2_t,imgs1_np,imgs2_np,positives_coarse


    def train(self):
        self.net.train()
        failures = 0

        with tqdm.tqdm(total=self.steps, initial=self.current_step) as pbar:
            # Replaced for loop with while loop to count only successful completion steps
            while self.current_step < self.steps:
                temp = self.generate_train_data()
                if temp is None:
                    failures += 1
                    if failures > self.config.max_load_failures:
                        raise RuntimeError(f"Data loading failed {failures} times in a row.")
                    continue
                
                imgs1_t, imgs2_t, imgs1_np, imgs2_np, positives_coarse = temp

                is_corrupted = False
                for p in positives_coarse:
                    if len(p) < 30:
                        is_corrupted = True

                if is_corrupted:
                    failures += 1
                    if failures > self.config.max_load_failures:
                        raise RuntimeError(f"Corrupted batches received {failures} times in a row.")
                    continue
                
                failures = 0  # Reset on successful batch load
                
                # --- Teacher Extraction (No Autograd Tracking) ---
                with torch.no_grad():
                    alike_kpts1, alike_kpts2 = [], []
                    DA_normals1, DA_normals2 = [], []
                    
                    for b in range(len(imgs1_np)):
                        _, megadepth_norm1 = self.depth_net.extract(imgs1_np[b])
                        _, megadepth_norm2 = self.depth_net.extract(imgs2_np[b])
                        DA_normals1.append(megadepth_norm1)
                        DA_normals2.append(megadepth_norm2)
                        
                    for b in range(imgs1_t.shape[0]):
                        # Avoid tensor tracking to offload host memory copy latency
                        img1 = imgs1_t[b].permute(1,2,0).expand(-1,-1,3).detach().cpu().numpy() * 255
                        img2 = imgs2_t[b].permute(1,2,0).expand(-1,-1,3).detach().cpu().numpy() * 255
                        alike_kpt1 = torch.tensor(self.alike_net.extract_alike_kpts(img1), device=self.dev)
                        alike_kpt2 = torch.tensor(self.alike_net.extract_alike_kpts(img2), device=self.dev)
                        alike_kpts1.append(alike_kpt1)
                        alike_kpts2.append(alike_kpt2)

                # --- Student Forward Pass w/ AMP context ---
                with torch.cuda.amp.autocast(enabled=self.config.amp):
                    feats1, kpts1, normals1, normals_feat1 = self.net.forward1(imgs1_t)
                    feats2, kpts2, normals2, normals_feat2 = self.net.forward1(imgs2_t)
                    
                    coordinates, fb_coordinates = None, None
                    fb_feats1, fb_feats2 = [], []
                    
                    if self.use_coord_loss:
                        coordinates, fb_coordinates = [], []
                    
                    for b in range(feats1.shape[0]):
                        fb_feat1 = self.net.forward2(feats1[b].unsqueeze(0), normals_feat1[b].unsqueeze(0))
                        fb_feat2 = self.net.forward2(feats2[b].unsqueeze(0), normals_feat2[b].unsqueeze(0))
                        fb_feats1.append(fb_feat1.unsqueeze(0))
                        fb_feats2.append(fb_feat2.unsqueeze(0))
                        
                        # Only compute fine_matcher outputs when coordinate loss is explicitly activated
                        if self.use_coord_loss:
                            feat1 = feats1[b].permute(1,2,0).reshape(-1, feats1.shape[1])
                            feat2 = feats2[b].permute(1,2,0).reshape(-1, feats2.shape[1])
                            coordinates.append(self.net.fine_matcher(torch.cat([feat1, feat2], dim=-1)))
                            
                            fb_f1 = fb_feat1.squeeze(0).permute(1,2,0).reshape(-1, fb_feat1.shape[1])
                            fb_f2 = fb_feat2.squeeze(0).permute(1,2,0).reshape(-1, fb_feat2.shape[1])
                            fb_coordinates.append(self.net.fine_matcher(torch.cat([fb_f1, fb_f2], dim=-1)))

                    fb_feats1 = torch.cat(fb_feats1, dim=0).reshape(feats1.shape[0], feats1.shape[2], feats1.shape[3], -1).permute(0,3,1,2)
                    fb_feats2 = torch.cat(fb_feats2, dim=0).reshape(feats2.shape[0], feats2.shape[2], feats2.shape[3], -1).permute(0,3,1,2)
                    
                    if self.use_coord_loss:
                        coordinates = torch.cat(coordinates, dim=0).reshape(feats1.shape[0], feats1.shape[2], feats1.shape[3], -1).permute(0,3,1,2)
                        fb_coordinates = torch.cat(fb_coordinates, dim=0).reshape(feats1.shape[0], feats1.shape[2], feats1.shape[3], -1).permute(0,3,1,2)

                    loss_info = self.loss_fn(
                        feats1, fb_feats1, kpts1, normals1,
                        feats2, fb_feats2, kpts2, normals2,
                        positives_coarse,
                        coordinates, fb_coordinates,
                        alike_kpts1, alike_kpts2,
                        DA_normals1, DA_normals2,
                        self.megadepth_batch_size, self.coco_batch_size
                    )
                    
                    loss_items = [loss_info['loss_fb_descs'].unsqueeze(0), loss_info['loss_kpts'].unsqueeze(0), loss_info['loss_normals'].unsqueeze(0)]
                    if self.use_coord_loss:
                        loss_items.append(loss_info['loss_fb_coordinates'].unsqueeze(0))
                    
                    loss = torch.cat(loss_items, -1).mean()

                # --- Scaled Backward Pass ---
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.opt)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 1.)
                
                self.scaler.step(self.opt)
                self.scaler.update()
                
                # set_to_none limits overwriting blocks to zero
                self.opt.zero_grad(set_to_none=True)
                self.scheduler.step()
                
                self.current_step += 1
                pbar.update(1)

                # --- Decoupled Logging / Sync Barriers ---
                if self.current_step % self.config.log_every == 0 or self.current_step == self.steps:
                    l_descs, acc_c = loss_info['loss_descs'].item(), loss_info['acc_coarse']
                    l_fb_descs, acc_fb_c = loss_info['loss_fb_descs'].item(), loss_info['acc_fb_coarse']
                    l_kpts, acc_k = loss_info['loss_kpts'].item(), loss_info['acc_kpt']
                    l_norm = loss_info['loss_normals'].item()
                    
                    l_coords = loss_info['loss_coordinates'].item() if self.use_coord_loss else 0.0
                    acc_coords = loss_info['acc_coordinates'] if self.use_coord_loss else 0.0
                    l_fb_coords = loss_info['loss_fb_coordinates'].item() if self.use_coord_loss else 0.0
                    acc_fb_coords = loss_info['acc_fb_coordinates'] if self.use_coord_loss else 0.0
                    
                    pbar.set_description(
                        f"Step: {self.current_step}/{self.steps} "
                        f"Loss: {loss.item():.4f} "
                        f"L_descs: {l_descs:.3f} Acc_c: {acc_c:.3f} "
                        f"L_fb_descs: {l_fb_descs:.3f} Acc_fb_c: {acc_fb_c:.3f} "
                        f"L_kpts: {l_kpts:.3f} Acc_k: {acc_k:.3f} "
                        f"L_norms: {l_norm:.3f}"
                    )
                    
                    self.writer.add_scalar('Step/current', self.current_step, self.current_step)
                    self.writer.add_scalar('Loss/total', loss.item(), self.current_step)
                    self.writer.add_scalar('Loss/descs', l_descs, self.current_step)
                    self.writer.add_scalar('Loss/fb_descs', l_fb_descs, self.current_step)
                    self.writer.add_scalar('Loss/kpts', l_kpts, self.current_step)
                    self.writer.add_scalar('Loss/normals', l_norm, self.current_step)
                    self.writer.add_scalar('Accuracy/acc_coarse', acc_c, self.current_step)
                    self.writer.add_scalar('Accuracy/acc_fb_coarse', acc_fb_c, self.current_step)
                    
                    if self.use_coord_loss:
                        self.writer.add_scalar('Loss/coordinates', l_coords, self.current_step)
                        self.writer.add_scalar('Loss/fb_coordinates', l_fb_coords, self.current_step)
                        self.writer.add_scalar('Accuracy/acc_coordinates', acc_coords, self.current_step)
                        self.writer.add_scalar('Accuracy/acc_fb_coordinates', acc_fb_coords, self.current_step)

                # --- Atomic Checkpoint Saves ---
                if self.current_step % self.save_ckpt_every == 0:
                    print(f'saving iter {self.current_step}')
                    
                    save_dict = {
                        "model": self.net.state_dict(),
                        "optimizer": self.opt.state_dict(),
                        "scheduler": self.scheduler.state_dict(),
                        "step": self.current_step,
                        "current step": self.current_step
                    }
                    
                    if self.config.amp:
                        save_dict["scaler"] = self.scaler.state_dict()
                        
                    tmp_path = self.ckpt_save_path + f'/{self.model_name}_{self.current_step}.pth.tmp'
                    final_path = self.ckpt_save_path + f'/{self.model_name}_{self.current_step}.pth'
                    
                    torch.save(save_dict, tmp_path)
                    os.replace(tmp_path, final_path)

                if self.dry_run:
                    break


if __name__ == '__main__':
    setproctitle.setproctitle(config.name)
    trainer = Trainer(config)
    trainer.train()