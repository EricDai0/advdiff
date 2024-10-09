import sys
import os 
import random
import argparse

sys.path.append(".")
sys.path.append('./taming-transformers')

import torch
from omegaconf import OmegaConf

from ldm.util import instantiate_from_config

from ldm.models.diffusion.ddim_adv import DDIMSampler
from torchvision.models import resnet50, ResNet50_Weights

from torchvision.utils import save_image
from torch.backends import cudnn

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid
    
parser = argparse.ArgumentParser()

parser.add_argument('--batch-size', type=int, default=6)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--scale', type=float, default=3.0)
parser.add_argument('--ddim-steps', type=int, default=200)
parser.add_argument('--ddim-eta', type=float, default=0.0)
parser.add_argument('--K', type=int, default=5)
parser.add_argument('--s', type=float, default=1.0)
parser.add_argument('--a', type=float, default=0.5)
parser.add_argument('--save-dir', type=str, default='advdiff/')
args = parser.parse_args()

def load_model_from_config(config, ckpt):
    print(f"Loading model from {ckpt}")
    pl_sd = torch.load(ckpt)#, map_location="cpu")
    sd = pl_sd["state_dict"]
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    model.cuda()
    model.eval()
    return model


def get_model():
    config = OmegaConf.load("configs/latent-diffusion/cin256-v2.yaml")  
    model = load_model_from_config(config, "models/ldm/cin256-v2/model.ckpt")
    return model

def main():
    cudnn.benchmark = False
    cudnn.deterministic = True
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    
    model = get_model()
    vic_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(model.device)
    vic_model.eval()
    sampler = DDIMSampler(model, vic_model=vic_model)

    classes =  np.arange(1000)
    n_samples_per_class = args.batch_size

    weights = ResNet50_Weights.DEFAULT
    preprocess = weights.transforms()

    ddim_steps = args.ddim_steps
    ddim_eta = args.ddim_eta
    scale = args.scale   # for unconditional guidance


    all_samples = list()
    all_labels = list()

    with torch.no_grad():
        with model.ema_scope():
            uc = model.get_learned_conditioning(
                {model.cond_stage_key: torch.tensor(n_samples_per_class*[1000]).to(model.device)}
                )

            for class_label in classes:
                print(f"rendering {n_samples_per_class} examples of class '{class_label}' in {ddim_steps} steps and using s={scale:.2f}.")
                xc = torch.tensor(n_samples_per_class*[class_label])
                c = model.get_learned_conditioning({model.cond_stage_key: xc.to(model.device)})

                samples_ddim, _ = sampler.sample(S=ddim_steps,
                                                 conditioning=c,
                                                 batch_size=n_samples_per_class,
                                                 shape=[3, 64, 64],
                                                 verbose=False,
                                                 unconditional_guidance_scale=scale,
                                                 unconditional_conditioning=uc, 
                                                 eta=ddim_eta,
                                                 label=xc.to(model.device),
                                                 K=args.K,s=args.s,a=args.a)

                x_samples_ddim = model.decode_first_stage(samples_ddim)
                x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                             min=0.0, max=1.0)
                all_samples.append(x_samples_ddim)
                all_labels.append(xc)

    img = torch.cat(all_samples, 0)
    labels = torch.cat(all_labels, 0)

    save_img = img.permute(0,2,3,1)

    np.savez(os.path.join(args.save_dir, 'AdvDiff.npz'), save_img.detach().cpu().numpy(), labels.detach().cpu().numpy())

            
if __name__ == '__main__':
    main()
