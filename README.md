# GR-KAT: Group-Rational Kolmogorov–Arnold Transformers for MRI Brain Tumor Classification

This repository provides an extended implementation of **Kolmogorov–Arnold Transformers (KAT)** adapted for **MRI brain tumor classification** using configurable **group-wise rational nonlinearities**.

The implementation supports controlled ablation experiments and reproducible training on medical imaging datasets.

---

# 🚀 Key Features

- MRI Brain Tumor Classification (4 classes)
- Group-wise Rational Nonlinearities
- Rational-order configuration **(m,n) = (5,4)**
- Template-based initialization (GELU / Swish style)
- Fully learnable nonlinear mappings
- Reproducible training pipeline
- Modular architecture for research experimentation

---

# 🧠 Method Overview

This work extends **Kolmogorov–Arnold Transformers (KAT)** by introducing configurable **group-wise rational nonlinear functions**.

The rational activation is defined as:

f(u) =  
( Σ(i=0→m) aᵢ uⁱ ) /  
( 1 + | Σ(j=1→n) bⱼ uʲ | )

Unlike fixed activations (e.g., GELU or Swish), rational nonlinearities:

- Learn their functional shape during training  
- Adapt curvature dynamically  
- Improve feature representation flexibility  
- Enable richer modeling of MRI intensity patterns  

---

# 📂 Repository Structure

```
KAT_analysis_code/
│
├── rational_kat_cu/        # Original rational kernels
├── rational_kat_cu_m43/    # Rational kernel (m=4, n=3)
│
├── scripts/                # Training utilities
├── tools/                  # Helper utilities
│
├── katransformer.py        # Transformer backbone
├── train.py                # Training script
├── validate.py             # Validation script
│
├── dist_train.sh           # Distributed launcher
├── run_remaining.sh
├── resume_remaining.sh
│
└── README.md
```

---

# ⚡ Rational CUDA/Triton Backend

This implementation relies on the **rational_kat_cu** CUDA/Triton extension for efficient computation of group-wise rational nonlinearities.

Official Repository:

https://github.com/Adamdad/rational_kat_cu

This backend enables fast forward and backward computation of rational activation functions used in Kolmogorov–Arnold Transformers.

---

# 🔧 Installing rational_kat_cu

Clone and install the CUDA extension:

```bash
git clone https://github.com/Adamdad/rational_kat_cu.git
cd rational_kat_cu
pip install -e .
```

---

# 📌 Dataset Structure

Expected dataset format:

```
Brain_tumor/
│
├── train/
│   ├── glioma/
│   ├── meningioma/
│   ├── pituitary/
│   └── no_tumor/
│
└── test/
    ├── glioma/
    ├── meningioma/
    ├── pituitary/
    └── no_tumor/
```

---

# 📚 Citation

If you use this repository, please cite:

## Original KAT Work

```bibtex
@article{yang2024kat,
  title={Kolmogorov–Arnold Transformers},
  author={Yang, Xingyi},
  year={2024}
}
```

---

# 🙏 Acknowledgment

This repository builds upon:

Kolmogorov–Arnold Transformers (KAT)  
https://github.com/Adamdad/kat  

and uses the CUDA/Triton rational activation backend:

https://github.com/Adamdad/rational_kat_cu  

We thank the original authors for making their implementations publicly available.

---

# ⚖️ License

This project is based on code licensed under the **MIT License**.

Original License:

MIT License  
Copyright (c) 2024 Xingyi Yang  

All modifications:

Copyright (c) 2026  
Gurram Harshamanya Thilak
