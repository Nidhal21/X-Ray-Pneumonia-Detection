# X-Ray-Pneumonia-Detection with DenseNet121

A deep learning project to classify chest X-rays as "Pneumonia" or "Normal" using transfer learning with DenseNet121.

## Overview
- **Goal**: Automate pneumonia detection to assist radiologists.
- **Dataset**: [Kaggle Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia) - 5,216 training, 16 validation, 624 test images.
- **Tools**: TensorFlow, Keras, OpenCV, NumPy, Matplotlib, Kaggle GPUs.

## Model
- **Base**: DenseNet121 (pre-trained on ImageNet).
- **Custom Head**: Dense layers (4608, 1152, 2) with Dropout(0.2).
- **Approach**:
  1. **Initial Training**: Froze all base layers, trained the head (Val Acc: 75%, Test Acc: 79%).
  2. **Fine-Tuning**: Unfroze `conv5_block16_1_conv` onward 
- **Accuracy**: 
  - Model 1: 79% (test), 75% (val).
  - Model 2:86.5% (test), 93.75% (val).

## Installation
1. Clone the repo:
   ```bash
   git clone  https://github.com/Nidhal21/X-Ray-Pneumonia-Detection.git
   cd X-Ray-Pneumonia-Detection
