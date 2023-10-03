# AdvGCGAN
 
The code repository for our paper AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models [arXiv](https://arxiv.org/abs/2307.12499)

# Installation

This repository is based on the offical code from [Latent Diffusion Models](https://github.com/CompVis/latent-diffusion). 

1. Set up environments for the codes. Details please refer to the original Github code.

   ```shell
   conda env create -f environment.yaml
   conda activate ldm_adv
   ```
2. Download ImageNet checkpoint and save to "models/ldm/cin256-v2/model.ckpt"

# Usage

Currently, a simple reference code can be used with advdiff.py. Please refer to the ddim_adv.py for possible code transfer. A complete code will be implemented soon.


# Reference

Please cite our paper if you found any helpful information:


    @article{dai2023advdiff,
      title={AdvDiff: Generating Unrestricted Adversarial Examples using Diffusion Models},
      author={Dai, Xuelong and Liang, Kaisheng and Xiao, Bin},
      journal={arXiv preprint arXiv:2307.12499},
      year={2023}
    }
