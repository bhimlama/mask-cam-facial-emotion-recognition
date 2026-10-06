# Mask-CAM: Self-Supervised Masking for Interpretable and Robust Facial Emotion Recognition

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23132270.svg)](https://doi.org/10.5281/zenodo.23132270)

**Bhim Lama**  
M.Sc. Thesis, Tribhuvan University, 2026  
Supervisor: Prof. Dr. Subarna Shakya

---

## Overview

Mask-CAM is a lightweight, intrinsically interpretable convolutional network for facial emotion recognition (FER). It learns a self-supervised soft mask that suppresses background regions and focuses on discriminative facial features — without any external mask labels. The model addresses two common problems in FER:

- Class imbalance — rare emotions like disgust are almost never detected by standard models
- Lack of interpretability — black-box models give no evidence of where they looked

## Key Results

| Metric | BaselineCNN | CAMCNN | Mask-CAM |
|---|---|---|---|
| FERPlus Accuracy | 79.41% | 79.16% | 78.85% |
| FERPlus Macro F1 | 60.08% | 63.87% | 65.60% |
| RAF-DB Accuracy  | 66.43% | 67.28% | 67.50% |
| RAF-DB Macro F1  | 43.29% | 46.73% | 46.71% |
| Parameters       | 2.84 M | 0.47 M | 0.48 M |

- Best macro F1 on FERPlus among compared models
- Only model to detect disgust on the cross-dataset RAF-DB benchmark (F1 = 0.05)
- Overlay accuracy: 99.3% (FERPlus), 100% (RAF-DB)
- 6x fewer parameters than the black-box baseline
- Smallest cross-dataset accuracy drop (11.89 pp under TTA)

## Architecture

Mask-CAM extends a standard CAM network with a self-supervised soft mask generator. The mask is applied element-wise to the backbone's feature maps before the classifier. It is trained with a composite loss:

    L_total = L_CE + lambda * mean(mask) + beta * TV(mask)

where:

- L_CE = cross-entropy with label smoothing
- mean(mask) = sparsity penalty (pushes the mask to suppress irrelevant regions)
- TV(mask) = total-variation smoothness penalty (encourages contiguous, blob-like masks)
- lambda and beta are loss weights (see configs/hyperparameters.yaml)

The lambda weight is linearly warmed up over the first 10 epochs to prevent mask collapse.

## Repository Structure

    ├── paper/
    │   └── Thesis_Report.pdf      # Full thesis
    ├── src/
    │   ├── models/                # Model architectures
    │   │   ├── backbone.py        #   Shared ImprovedBackbone
    │   │   ├── baseline_cnn.py    #   BaselineCNN (black-box)
    │   │   ├── cam_cnn.py         #   CAMCNN (standard CAM)
    │   │   └── mask_cam.py        #   MaskCAM (main contribution)
    │   ├── data/                  # Dataset loaders
    │   │   ├── ferplus.py         #   FERPlus dataset
    │   │   ├── rafdb.py           #   RAF-DB (cross-dataset)
    │   │   ├── transforms.py      #   Data augmentation
    │   │   └── utils.py           #   TensorDataset
    │   ├── losses.py              # Composite loss + lambda warm-up
    │   ├── train.py               # Training loops
    │   ├── training_utils.py      # MixUp helpers
    │   ├── metrics.py             # Accuracy, F1, TTA, McNemar
    │   ├── efficiency.py          # Parameter count, MACs, inference time
    │   ├── landmarks.py           # Facial landmark utilities
    │   └── visualization.py       # CAM and mask visualization
    ├── configs/
    │   └── hyperparameters.yaml   # Training configuration (Table 5)
    ├── requirements.txt
    └── README.md

## Where to Start Reading

If you're new to this project, the two most important files are:

1. src/models/mask_cam.py - the core model architecture (~60 lines)
2. src/losses.py - the composite loss and warm-up schedule (~85 lines)

Together they contain the entire novel contribution.

## Datasets

- FERPlus - extension of FER2013 with crowd-sourced labels (Barsoum et al., 2016)
- RAF-DB - Real-world Affective Faces Database (Li et al., 2017)

Both datasets are publicly available. This repository does not redistribute them.

## Setup

    git clone https://github.com/bhimlama/mask-cam-facial-emotion-recognition.git
    cd mask-cam-facial-emotion-recognition
    pip install -r requirements.txt

## Citation

If you use this work, please cite:

    @mastersthesis{lama2026maskcam,
      title={Self-Supervised Masking for Interpretable and Robust Facial Emotion Recognition},
      author={Lama, Bhim},
      year={2026},
      month={August},
      school={Tribhuvan University, Institute of Science and Technology},
      type={Master's thesis},
      doi={10.5281/zenodo.23132270}
    }


## Contact

- Email: lbhim81577@gmail.com
- LinkedIn: https://www.linkedin.com/in/bhim-lama-395289273/
- GitHub: https://github.com/bhimlama

## License

MIT — see LICENSE for details.
