# PanMatch: Unleashing the Potential of Large Vision Models for Unified Matching Models

---

## 📜 [Paper Link](https://arxiv.org/pdf/2507.08400)

---

## 📢 Project Status

We are actively organizing and cleaning up the full codebase (training, inference, evaluation).  
The repository will be **continuously updated** in the coming weeks — please stay tuned for:

- 🛠️ Full training scripts
- 📦 Data preparation scripts
- 📈 Evaluation pipelines

---

## 📝 Updates
- Demo code and pre-trained weights are released. (07/22/2025)

---

## 🔗 Pre-trained Checkpoints

You can find the released pre-trained checkpoints here:  
- PanMatch Checkpoints  [(Google Drive)](https://drive.google.com/file/d/18pV4RzO2_AdKCrQwSgHf57tfD3xB9S77/view?usp=drive_link)

- DINOv2 Checkpoints [https://github.com/facebookresearch/dinov2?tab=readme-ov-file](https://dl.fbaipublicfiles.com/dinov2/dinov2_vitg14/dinov2_vitg14_reg4_pretrain.pth)

Place the PanMatch weights in the `/ckpt` folder, and place the DINOv2 weights in the `/src/foundation_model_weights` directory.

---

## 🚀 Quick Start

### installation

```
conda create -n PanMatch python=3.11
conda activate PanMatch
conda install pytorch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 pytorch-cuda=11.8 -c pytorch -c nvidia
conda install tqdm matplotlib scikit-image
pip install timm==0.4.12
pip install opencv-python tensorboard imath einops h5py kornia poselib loguru yacs kornia_moons
pip install "numpy<2"
```

### inference
Modify the `demo_options.json` in `src/Options`, and then run
```
cd src
sh demo.sh
```

## ✅ TODO

- [ ] **Data preparesion and pre-training code**  
  (including dataset preprocessing, augmentation, and full training pipeline)

- [ ] **Inference and evaluation code on multiple tasks**  
  (standardized pipelines for benchmarks and visualization)

---

Thank you for your interest and support! ⭐️  
Feel free to open an issue if you have any questions or suggestions.

---


## 🌹 Acknowledgements
Part of the code is adapted from previous works: 
 - [FormerStereo](https://github.com/zhangyj85/FormerStereo_release) (code base)
 - [FlowFormer](https://github.com/drinkingcoder/FlowFormer-Official) (baseline)
 - [WxBS](https://github.com/ducha-aiki/wxbs-descriptors-benchmark) (visulization for feature matching)

We thank all the authors for their awesome repos.



## ✏️ Citation
If you find the code helpful in your research, please cite:

```
@article{zhang2025panmatch,
  title={{PanMatch}: Unleashing the Potential of Large Vision Models for Unified Matching Models},
  author={Zhang, Yongjian and Wang, Longguang and Li, Kunhong and Zhang, Ye and Wang, Yun and Lin, Liang and Guo, Yulan},
  journal={arXiv preprint arXiv:2507.08400},
  year={2025}
}
```
