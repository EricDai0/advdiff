# AdvDiff
 
The code repository for our paper AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models [arXiv](https://arxiv.org/abs/2307.12499)

# Installation

This repository is based on the offical code from [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion). 

1. Set up environments for the codes. Details please refer to the original Github code.

   ```shell
   conda env create -f environment.yaml
   conda activate ldm_adv
   ```
2. Download ImageNet checkpoint [LDM] (https://ommer-lab.com/files/latent-diffusion/nitro/cin/model.ckpt) and save to "models/ldm/cin256-v2/model.ckpt"

# Usage

Simply run advdiff.py to perform attacks against the official PyTorch ResNet50 model. You can modify the attack parameter at the advdiff.py.
```
python advdiff.py 
```

# Reference

Please cite our paper if you found any helpful information:


    @article{dai2023advdiff,
      title={AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models},
      author={Dai, Xuelong and Liang, Kaisheng and Xiao, Bin},
      journal={arXiv preprint arXiv:2307.12499},
      year={2023}
    }
