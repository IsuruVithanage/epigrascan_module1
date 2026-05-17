# Epigrascan - Module 01 📜🔍

## Overview

**Epigrascan** is an AI-Driven Decision Support System designed for the Character Extraction, Probabilistic Restoration, and Segmentation of Early Brahmi Estampages.

Deciphering ancient Early Brahmi stone inscriptions from paper rubbings (estampages) is highly subjective and prone to human cognitive bias. Traditional monolithic AI models fail on these artifacts due to severe physical weathering, partial character erosion, and continuous spaceless writing (*scriptio continua*). 

Epigrascan solves this via a **decoupled, three-stage decision support system**. This repository module encompasses the first two critical stages: specialized computer vision to extract characters from extreme stone noise, and a deep learning classifier to identify them.

## 🏗️ Project Architecture

### 1. Stage 1: Segmentation (Image Cleaning & Cropping)
Stone rubbings are heavily unstructured and noisy. This stage isolates the text from the stone texture using advanced morphological and mathematical techniques.
* **Key Techniques:** Non-Local Means (NLM) denoising, Bilateral filtering, polynomial baseline tracking, and Connected Component Analysis (CCA).
* **Goal:** Intelligently separate the rock background from the ink and cleanly crop out individual characters.

### 2. Stage 2: Classification (Deep Learning Inference)
Once characters are cleanly isolated, they are passed to a lightweight convolutional neural network.
* **Model:** MobileNetV2, classifying 368 distinct Brahmi classes (consonants, vowels, and compounds).
* **Curriculum Learning:** The model training dynamically injects heavy synthetic noise (simulating stone cracks, erosion, and elastic warping from bad rubbings) at later epochs to ensure robust, real-world performance.
* **Output:** JSON files containing probabilistic transliteration sequences, confidence scores, and raw softmax outputs.

---

## 📂 Folder Structure

The project has been organized into an industry-standard format to separate active production code from raw data and legacy research scripts.

```text
epigrascan/
│
├── src/                                # 🚀 Active Production Codebase
│   ├── 01_segmentation/                # Image Cleaning & Cropping
│   │   ├── darken2.py                  # Main segmentation entry point
│   │   └── modules/                    # Specialized OpenCV R&D modules
│   │       ├── baseline.py
│   │       ├── calibration.py
│   │       ├── counting.py
│   │       ├── preprocessing.py
│   │       ├── segmentation.py
│   │       └── visualization.py
│   │
│   ├── 02_classification/              # Deep Learning Inference & Training
│   │   ├── train_classifier.py         # MobileNetV2 curriculum training script
│   │   ├── inference.py                # Final robust inference pipeline
│   │   └── utils/                      # Helper scripts (label mappings, setup testing)
│   │
│   └── 03_restoration/                 # Probabilistic Word Spacing & Layout
│       └── check_json.py               # Validates final output data
│
├── data/                               # Datasets (Raw, Processed, Synthetic)
├── models/                             # Saved model weights (.pth)
├── outputs/                            # Results, logs, and extracted images
│
├── archive_deprecated/                 # 🗄️ Old code (Safely quarantined)
│   ├── v1_monolithic_pipeline/         # Original monolithic scripts
│   ├── v2_modular_pipeline/            # Early modular attempts
│   └── experimental_scripts/           # Hardcoded/batch scripts
│
├── config.yaml                         # Global configuration (paths, model hyperparams)
├── label_mapping.json                  # Class mappings (Class ID -> Character)
└── requirements.txt                    # Project dependencies
```

---

## ⚙️ Installation & Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/YourUsername/epigrascan.git
   cd epigrascan
   ```

2. **Set up a virtual environment (Optional but Recommended):**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```
   *Note: Ensure you have a CUDA-enabled GPU and the correct PyTorch version if you plan on training or running fast inference.*

4. **Verify PyTorch/CUDA Setup:**
   ```bash
   python src/02_classification/utils/test_setup.py
   ```

---

## 🚀 Usage

All configurations (dataset paths, model hyperparameters, training settings) are managed centrally in `config.yaml`. 

### 1. Segmentation
Run the segmentation pipeline to clean and extract character crops from raw estampages.
```bash
python src/01_segmentation/darken2.py
```

### 2. Training the Classifier
Train the MobileNetV2 model using the curriculum learning schedule defined in `config.yaml`.
```bash
python src/02_classification/train_classifier.py
```

### 3. Inference
Run the inference pipeline to classify previously segmented characters.
```bash
python src/02_classification/inference.py
```

### 4. Restoration
Validate outputs and reconstruct the document space.
```bash
python src/03_restoration/check_json.py
```

---

## 🧠 Model Configuration (`config.yaml`)

- **Architecture:** `mobilenet_v2`
- **Classes:** 368 (33 consonants, 10 vowels, 325 compounds)
- **Input Size:** `224x224`
- **Curriculum Learning:** Introduces heavy rock damage augmentations at epoch 8 to improve generalization on real-world damaged estampages.

---

## ⚠️ Notes on Legacy Code
All deprecated code (monolithic v1, older backups) has been moved to the `archive_deprecated/` directory. If you are a new contributor, **do not** use or import files from this directory unless retrieving old experimental logic. All active development should take place inside `src/`.
