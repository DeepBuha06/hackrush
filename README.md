# AI-vs-Real Image Classification using LeViT-192

This project was built during the **HackRush Hackathon** organized by the Academic Council, IIT Gandhinagar. It addresses **Problem 3B: Image Classification**, where the goal was to classify images as either **AI-generated** or **Real** using a deep learning model.

---

## Problem Statement

With the rise of AI-generated imagery (deepfakes, generative art, etc.), it's critical to distinguish between real and AI-generated images. This project uses a **LeViT-192 Vision Transformer** model to tackle this binary classification task.

---

## Dataset

- **Source:** Provided on **Kaggle** during the hackathon.
- **Size:** Contains over **60,000 labeled images** (AI vs. Real).
- All images were resized to **224×224** and normalized.

---

## Model

- Model: [`facebook/levit-192`](https://huggingface.co/facebook/levit-192) (Vision Transformer via TIMM)
- Trained **from scratch** on the provided dataset.
- Achieved **85.26% accuracy** on the test set.

---

## Methodology

### Preprocessing
- Resize to 224×224
- Normalize using `[0.5, 0.5, 0.5]` mean and std

### Data Augmentation
- Random rotations
- Horizontal flips
- Color jitter
- Random cropping

### Training
- Loss: Binary Cross-Entropy
- Optimizer: Adam (lr = 1e-4)
- Trained for multiple epochs with model checkpoints saved

### Explainability
- Used **Grad-CAM** to visualize what parts of the image the model focuses on while making predictions.

---

## Usage

- Open and run the `problem_3b.ipynb` notebook.
- The notebook contains all steps from data loading, model training, evaluation, Grad-CAM visualization, and inference on new images.
- You can use the trained model weights (`ai_vs_real_levit192_model.pth`) to perform inference on your own images by following the inference cells inside the notebook.

---

## Results

- Achieved an accuracy of 85.26% on the test set.
- Grad-CAM visualizations help interpret which parts of the images the model uses to make decisions.

---

## Potential Improvements

- Adding Noise Perturbation Regularization (NPR) and Frequency Noise during training could improve robustness and generalization.
- Creating a standalone inference script to simplify deployment outside the notebook environment.

---

## Report

- See the detailed `report_3b.pdf` for more information on the methodology, experiments, and visualizations.

---

## Acknowledgements

- Dataset provided via Kaggle during the hackathon.
- LeViT model from the Facebook AI Research and implemented via TIMM.
- Grad-CAM implementation for model explainability.

---

## Repository Structure

```plaintext
├── ai_vs_real_levit192_model.pth    # Final trained model weights
├── best_model.pth                   # Checkpoint with best validation accuracy
├── config.py                       # Configuration file for hyperparameters and paths
├── dataset.py                      # Custom dataset class and data loading code
├── problem_3b.ipynb                # Main notebook for training, inference, and Grad-CAM visualization
├── submission (6).csv              # Prediction results for submission
├── report_3b.pdf                   # Detailed project report
