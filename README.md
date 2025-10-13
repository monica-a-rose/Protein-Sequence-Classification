# Fine-tuning ESM-2 for Protein Subcellular Localization

## Project Overview

This project demonstrates a complete end-to-end workflow for fine-tuning a pre-trained ESM-2 transformer model to predict protein subcellular localization (Cytoplasm vs. Nucleus). Using a dataset of 3,575 protein sequences from UniProt, this project covers data cleaning, exploratory data analysis, a robust training pipeline, and model evaluation.

The final model achieves a **ROC-AUC Score of 0.9184** and a **Test Accuracy of 85.3%**, showcasing the power of protein language models for bioinformatics tasks.

---

## Key Features

-   **Data Cleaning:** Systematic handling of duplicate sequences to prevent data leakage.
-   **Exploratory Data Analysis (EDA):** In-depth analysis of class balance and protein sequence length distribution to inform modeling strategy.
-   **Optimal Tokenization:** Determined a `max_length` of 482 based on the 95th percentile of tokenized lengths to balance performance and computational efficiency.
-   **Hugging Face Trainer:** Utilized the Hugging Face `Trainer` API for a robust and efficient training loop, including mixed-precision training (FP16).
-   **Model Evaluation:** Comprehensive model assessment including loss/accuracy curves, classification reports, confusion matrix, and ROC/PR curves.
-   **Hyperparameter Analysis:** Compared model performance with different learning rates (2e-5 vs. 1e-4) to analyze training dynamics and overfitting.

---

## Technologies and Libraries

-   **Core Libraries:** Python, PyTorch, Hugging Face (Transformers, Datasets, Accelerate), scikit-learn, pandas
-   **Bioinformatics:** `fair-esm`, `biopython`
-   **Visualization:** Matplotlib, Seaborn
-   **Environment:** Jupyter Notebook

---

## Data

The dataset (`uniprot_protein_data_assigment.csv`) is included in the `/Data` directory. It contains protein sequences from UniProt and their corresponding subcellular location labels (Nucleus or Cytoplasm).

---

## Results

The model fine-tuned with a learning rate of 2e-5 achieved the following performance on the hold-out test set:

| Metric        | Score  |
|---------------|--------|
| Test Accuracy | 85.31% |
| ROC-AUC Score | 0.9184 |
| F1-Score (Cytoplasm) | 0.8679 |
| F1-Score (Nucleus)   | 0.8346 |