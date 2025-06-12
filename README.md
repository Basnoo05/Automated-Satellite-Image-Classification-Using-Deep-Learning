Automated Satellite Image Classification Using Deep Learning
===========================================================

Project Overview
----------------

This project implements a deep learning system for the automated classification of satellite images into four terrain categories: **desert**, **green area**, **cloudy**, and **water**. The goal is to provide a scalable, production-ready solution for environmental monitoring, climate research, agriculture, and urban planning using advanced convolutional neural networks (CNNs) and transfer learning.

Key Features
------------

- **Dual Model Architecture:** Custom CNN and transfer learning (ResNet50V2) for comprehensive performance evaluation.
- **Robust Data Pipeline:** Balanced dataset of 5,631 high-resolution (128×128×3) satellite images, advanced data augmentation, and stratified sampling.
- **Production-Ready:** Includes preprocessing, training, evaluation, and visualisation pipelines for operational deployment.
- **High Accuracy:** Up to 97.52% validation accuracy with ResNet50V2, demonstrating state-of-the-art performance.

Dataset
-------

- **Size:** 5,631 images
- **Classes:** Water (1,500), Desert (1,407), Cloudy (1,224), Green Area (1,500)
- **Resolution:** 128×128×3 (RGB)
- **Split:** 80% training, 20% testing (stratified sampling)
- **Balanced Distribution:** Ensures robust model evaluation and fair comparison.

Model Architectures
-------------------
**Custom CNN**

- **Architecture Details:**
    - 4-block progressive design (filters: 32 → 64 → 128 → 256)
    - Each block: Conv2D, BatchNorm, MaxPool2D
    - Classification head: Dense(512), Dropout(0.5), Dense(256), Dropout(0.3), Dense(4)
    - Advanced regularisation: Batch normalisation and strategic dropout
- **Key Features:**
    - Domain-specific design for satellite imagery
    - Efficient and robust for resource-constrained environments

**ResNet50V2 (Transfer Learning)**

- **Architecture Details:**
    - Pre-trained on ImageNet with frozen base layers
    - Custom classification head: Dense(128), Dropout(0.5), Dense(4)
- **Key Features:**
    - Leverages sophisticated feature extraction from large-scale datasets
    - High accuracy and rapid convergence for satellite image classification
Training Configuration
---------------------

- **Optimizer:** Adam
- **Loss:** Categorical cross-entropy
- **Epochs:** 5 (for direct comparison)
- **Batch Size:** 32
- **Callbacks:** Early stopping, learning rate reduction
- **Metrics:** Accuracy, precision, recall, F1-score

Performance
-----------

**Custom CNN**

- **Validation Accuracy:** 83.94%
- **F1-Score (Average):** ~85%
- **Notable Strengths:**
    - Strong generalization (validation accuracy exceeds training)
    - Efficient and suitable for resource-constrained environments

**ResNet50V2 (Transfer Learning)**

- **Validation Accuracy:** 97.52%
- **F1-Score (Average):** ~97%
- **Notable Strengths:**
    - Rapid convergence and high accuracy
    - Robust feature extraction and minimal overfitting
Usage
-----

1. **Clone the Repository:**

   .. code-block:: bash

      git clone https://github.com/yourusername/satellite-image-classification.git
      cd satellite-image-classification

2. **Install Requirements:**

   .. code-block:: bash

      pip install -r requirements.txt

3. **Prepare Data:**

   - Organise images into folders by class (water, desert, cloudy, green_area).
   - Update the CSV file with image paths and labels.

4. **Run Preprocessing:**

   .. code-block:: bash

      python preprocess.py

5. **Train Models:**

   .. code-block:: bash

      python train_custom_cnn.py
      python train_resnet50v2.py

6. **Evaluate Models:**

   .. code-block:: bash

      python evaluate.py

7. **Visualize Results:**

   - Use provided scripts to plot training curves, confusion matrices, and sample predictions.

Results and Insights
--------------------

- **High Accuracy:** Both models achieve strong performance, with ResNet50V2 outperforming the custom CNN.
- **Robust Generalisation:** Validation accuracy matches or exceeds training, indicating reliable performance on unseen data.
- **Production Readiness:** The pipeline is suitable for operational deployment in environmental monitoring, agriculture, and disaster response.

Challenges and Solutions
------------------------

- **Computational Constraints:** Limited training epochs due to hardware; mitigated by optimized data loading and batch processing.
- **Data Quality Variations:** Addressed by advanced data augmentation and preprocessing.
- **Class Boundary Ambiguity:** Improved by multi-scale feature extraction and comprehensive evaluation.

Future Work
-----------

- **k-Fold Cross-Validation:** For more robust model evaluation.
- **Ensemble Methods:** Combine strengths of different models.
- **Multi-Modal Data:** Integrate elevation, weather, and temporal data for enhanced performance.
- **Edge/Cloud Deployment:** Optimize models for real-time, scalable analysis.


