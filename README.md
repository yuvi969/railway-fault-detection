# Railway Track Fault Detection using Deep Learning

This project presents a comparative study of **image classification** and **object detection** approaches for automated railway track fault detection using deep learning. The work focuses on understanding the practical limitations of classification-based methods and demonstrating the effectiveness and generalization of detection-based models.

---

## Project Overview

Railway track inspection is critical for safety but is traditionally manual, time-consuming, and error-prone. This project explores the use of deep learning to automate fault detection from images.

Two approaches are studied:
1. **Image Classification** – Detecting whether a fault exists in an image.
2. **Object Detection** – Detecting and localizing faults using bounding boxes.

---

##  Methodology

### 1. Classification Approach
- Model: **ResNet18**
- Task: Binary classification (Fault / No Fault)
- Dataset: Railway Track Fault Dataset
- Objective: Establish a baseline and analyze limitations of classification-based inspection.

### 2. Detection Approach
- Model: **YOLOv8n**
- Task: Fault localization using bounding boxes
- Datasets:
  - **Dataset A:** Railway Track Fault Dataset (small-scale)
  - **Dataset B:** Concrete Surface Crack Dataset (larger-scale, structural cracks)
- Objective: Compare detection performance and evaluate cross-dataset generalization.

---

## Key Experiments & Findings

###  Classification vs Detection
- Classification models can identify the presence of faults but **cannot localize them**.
- Detection models provide **spatial fault localization**, making them more suitable for real-world inspection.

###  Cross-Dataset Evaluation
- Detection performance improves significantly with larger and more diverse datasets.
- YOLOv8 demonstrates strong generalization across structurally similar crack datasets.

---

##  Results Summary

| Experiment | Dataset | Task | Key Outcome |
|---------|--------|------|------------|
| Experiment 1 | Railway Dataset | Classification | ~73% accuracy, limited practicality |
| Experiment 2 | Railway Dataset | Detection | mAP ~0.6 with localization |
| Experiment 3 | Crack Dataset | Detection | mAP ~0.9, strong generalization |

> Note: Exact metrics and plots are available in the `results/` directory.

---
## 📁 Repository Structure

Railway_Fault_Detection/
├── train_classification.py
├── README.md
├── .gitignore
├── results/
│ ├── datasetA_detection_results.png
│ └── datasetB_detection_results.png


> Datasets and trained weights are not included due to size constraints.

---

##  Tools & Technologies

- Python
- PyTorch
- YOLOv8 (Ultralytics)
- OpenCV
- Roboflow (annotation)
- Kaggle datasets

---

##  Future Work

- Real-time railway inspection using video streams
- Deployment on edge devices
- Multi-class defect detection
- Dataset expansion with real railway imagery

---

## License & Usage

This project is intended for **educational and research purposes**.  
Datasets used belong to their respective original authors.

---

## Acknowledgements

- Kaggle community for open datasets
- Ultralytics for YOLOv8
- Original dataset authors cited in the accompanying paper

---

##  Contact

For questions or collaboration, feel free to reach out via GitHub.



