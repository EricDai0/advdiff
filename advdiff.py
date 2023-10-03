import sys
import os 
import random

os.environ["CUDA_VISIBLE_DEVICES"]="7"

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


cudnn.benchmark = False
cudnn.deterministic = True
torch.manual_seed(0)
torch.cuda.manual_seed(0)
np.random.seed(0)
random.seed(0)

model = get_model()
vic_model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(model.device)
sampler = DDIMSampler(model, vic_model=vic_model)
vic_model.eval()

import numpy as np 
from PIL import Image
from einops import rearrange
from torchvision.utils import make_grid

classes = [448]
n_samples_per_class = 6

weights = ResNet50_Weights.DEFAULT
preprocess = weights.transforms()

ddim_steps = 200
ddim_eta = 0.0
scale = 3.0   # for unconditional guidance


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
                                             K=5,s=1.0,a=0.5)

            x_samples_ddim = model.decode_first_stage(samples_ddim)
            x_samples_ddim = torch.clamp((x_samples_ddim+1.0)/2.0, 
                                         min=0.0, max=1.0)
            all_samples.append(x_samples_ddim)
            all_labels.append(xc)

# display as grid
grid = torch.stack(all_samples, 0)
labels = torch.stack(all_labels, 0)

img = torch.flatten(grid, start_dim=0, end_dim=1)
labels = torch.flatten(labels, start_dim=0, end_dim=1)

    
save_img = img.permute(0,2,3,1)
print(save_img.shape)
print(labels.shape)

np.savez("AdvSa.npz", save_img.detach().cpu().numpy(), labels.detach().cpu().numpy())

print(grid.shape)

grid = rearrange(grid, 'n b c h w -> (n b) c h w')
grid = make_grid(grid, nrow=n_samples_per_class)

# to image
grid = 255. * rearrange(grid, 'c h w -> h w c').cpu().numpy()
x = Image.fromarray(grid.astype(np.uint8))
x.save("Result.jpg")


img_transformed = preprocess(img).to(model.device)

with torch.no_grad():
    output = vic_model(img_transformed)
# Tensor of shape 1000, with confidence scores over Imagenet's 1000 classes
rates, indices = output.sort(1, descending=True) 

for i in range(indices.shape[0]):
    print(indices[i][0])
    
save_image(img_transformed, "ImageNet_T.png", nrow=n_samples_per_class, normalize=True)

