
# Jailbreaking Deep Models
### NYU ECE-GY 7123 — Deep Learning Project 3 (Spring 2025)  
**Murali Peddi, Tejdeep Chippa, Vamsi Jonnakuti**

---

## 🧠 Overview

This project explores the vulnerability of deep image classifiers to adversarial examples. We attack a pretrained **ResNet-34** model on a 100-class ImageNet subset using three gradient-based methods:

- **FGSM**: Fast Gradient Sign Method (single-step ℓ∞-constrained)
- **PGD**: Projected Gradient Descent (multi-step, iterative)
- **Patch PGD**: Perturbs only a 32×32 region with higher local ε

Each attack is designed to be subtle yet effective, reducing classification accuracy with minimal visible distortion. All attacks operate under strict ℓ∞ or spatially bounded constraints.

---

## 📊 Results Summary

| **Attack Type**            | **Top-1 Acc.** | **Top-5 Acc.** | **Top-1 Drop (%)** |
|----------------------------|----------------|----------------|--------------------|
| Clean (no attack)          | 76.00%         | 94.20%         | 0.00               |
| FGSM (ε=0.02)              | 3.60%          | 20.80%         | 95.26              |
| PGD (ε=0.02, α=0.005)      | 0.00%          | 1.40%          | 100.00             |
| Patch PGD (ε=0.5, 32×32)   | 30.20%         | 67.60%         | 60.53              |

All attacks were verified to respect perturbation budgets with runtime ℓ∞ checks. PGD emerged as the most destructive, reducing Top-1 accuracy to 0%.

---

## 📁 Dataset & Preprocessing

- **Dataset**: 500 images across 100 ImageNet-1K classes
- **Format**: Each class stored under its WordNet synset folder
- **Labels**: Mapped using `labels_list.json`
- **Preprocessing**: Images resized to 224×224, normalized using:
  - Mean: `[0.485, 0.456, 0.406]`
  - Std: `[0.229, 0.224, 0.225]`

All attacks use raw pixel gradients for perturbation but normalized inputs for model inference.

---

## 🧠 Model Architecture & Attacks

- **Base Model**: `torchvision.models.resnet34(weights="IMAGENET1K_V1")`
- **Evaluation Mode**: Inference-only, GPU enabled

### Attack Implementations:

#### Task 1 — Clean Evaluation
- Baseline accuracy: Top-1: 76.00%, Top-5: 94.20%

####  Task 2 — FGSM (Fast Gradient Sign Method)
- ε = 0.02
- Single-step gradient in raw space
- Top-1 accuracy: 3.60%

#### Task 3 — PGD (Projected Gradient Descent)
- ε = 0.02, α = 0.005, steps = 10
- Iterative update with projection back to ε-ball
- Top-1 accuracy: 0.00%

#### Task 4 — Patch PGD (Localized)
- Perturb only 32×32 region, ε = 0.5
- Top-1 accuracy: 30.20%

All attacks include visualizations: original image, adversarial image, and amplified perturbation map.

---

## 🔧 Installation

```bash
pip install torch torchvision matplotlib tqdm Pillow
```

Or run directly on Google Colab (recommended): upload dataset zip and run dl-project-3.ipynb

---

## 🧪 Usage
1. Upload: TestDataSet.zip and extract it in the Colab environment.

2. Run: Execute all cells in dl-project-3.ipynb

3. Outputs:

   - AdversarialTestSet1/2/3: Perturbed images for each attack

   - Accuracy values printed for each task

   - Visualizations displayed for 5 sample images per attack

## ⚙️ Configuration (hardcoded in notebook)

-   FGSM: ε = 0.02
    
-   PGD: ε = 0.02, α = 0.005, steps = 10
    
-   Patch PGD: ε = 0.5 (in patch), patch size = 32×32, steps = 10

## 🌟 Conclusion

This project highlights how easily production-grade vision models can be "jailbroken" with small, targeted perturbations. We showed:

-   96–100% accuracy drop from FGSM and PGD with ε = 0.02
    
-   60% drop using patch-based attacks in just 32×32 regions
    

PGD proved most effective, while patch attacks revealed the impact of localized noise. These results support the need for adversarial training and model certification in real-world deployments.

## 📄 Project Report

The complete AAAI-formatted report is included as `DL_Project_3.pdf`. It covers:

-   Mathematical derivations
    
-   Evaluation metrics
    
-   Visual comparisons
    
-   Runtime validation of ε-constraints

## 🙏 Acknowledgments

-   PyTorch & TorchVision for pretrained models and utilities
    
-   ImageNet-1K validation subset and WordNet synset annotations
    
-   NYU Tandon ECE-GY 7123 (Spring 2025) for project guidance on adversarial robustness
    
-   ChatGPT for support with formatting, scripting, and report writingd
