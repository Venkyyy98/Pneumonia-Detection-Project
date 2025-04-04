# Pneumonia Detection: A Custom CNN and MobileNet Ensemble Approach

## 📌 Overview
This project focuses on detecting pneumonia from chest X-ray images using a deep learning ensemble model combining **Custom CNN and MobileNet** architectures. The model enhances pneumonia detection accuracy through **data augmentation, ensemble modeling, and fine-tuned hyperparameters**.

## 🏆 Key Features
- Hybrid **CNN-MobileNet** ensemble model.
- **97.63% F1-score** achieved on test data.
- Data preprocessing with **image augmentation**.
- **Grad-CAM visualization** for explainability.
- Implementation in **TensorFlow/Keras**.

## 📂 Dataset
- **Source**: [Chest X-Ray Images (Pneumonia) Dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)
- **Classes**: Normal vs. Pneumonia
- **Size**: ~5,863 images (JPEG)
- **Preprocessing**: Resizing to 224x224, normalization

## 🔬 Model Architecture
The project implements an **ensemble learning** approach by combining:
1. **Custom CNN** - Built with Conv2D, MaxPooling, BatchNorm, Dropout layers.
2. **MobileNetV2** - Lightweight pre-trained model for feature extraction.
3. **Ensemble Strategy** - Averaging predictions from both models.

## 📊 Results
| Metric     | Score |
|------------|-------|
| Accuracy   | 98.12% |
| Precision  | 97.5%  |
| Recall     | 97.8%  |
| F1-score   | 97.63% |

## 🔧 Installation & Usage
### 1️⃣ Setup
```bash
git clone https://github.com/Venkyyy_98/Pneumonia-Detection-Project.git
cd Pneumonia-Detection-CNN-MobileNet
pip install -r requirements.txt
