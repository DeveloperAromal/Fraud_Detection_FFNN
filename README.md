# Fraud Detection using Feed-Forward Neural Network (FFNN)

This repository implements a fraud detection system using a Feed-Forward Neural Network (FFNN) built with PyTorch.  
The project follows a modular and production-oriented structure, covering data preprocessing, model training, evaluation, k-fold validation, risk scoring, and API-based inference.

---

## Project Overview

Fraud detection is a critical problem in financial and transactional systems due to class imbalance and evolving fraud patterns.  
This project applies a supervised deep learning approach to classify transactions as fraudulent or legitimate using structured data.

The system includes:
- Data preprocessing and feature engineering
- Feed-Forward Neural Network model
- K-Fold cross-validation
- Model checkpointing
- Risk score computation
- ROC curve generation
- REST API for inference

---

## Model Details

- Model type: Feed-Forward Neural Network (FFNN)
- Framework: PyTorch
- Task: Binary classification
- Loss function: Binary Cross Entropy
- Optimizer: Adam
- Validation strategy: K-Fold Cross Validation

---

## Project Structure

```
fraud_detection_ffnn
├─ config
│  ├─ nn_config.py
│  └─ __pycache__
│     └─ nn_config.cpython-311.pyc
├─ data
│  ├─ processed
│  │  ├─ dataset.csv
│  │  └─ test_dataset.csv
│  └─ raw
│     └─ dataset.csv
├─ graphs
│  ├─ roc.py
│  └─ __pycache__
│     └─ roc.cpython-311.pyc
├─ main.py
├─ model
│  ├─ checkpoints
│  │  ├─ fold_1.pt
│  │  ├─ fold_2.pt
│  │  ├─ fold_3.pt
│  │  └─ fold_4.pt
│  ├─ predict.py
│  ├─ train.py
│  ├─ __init__.py
│  └─ __pycache__
│     ├─ predict.cpython-311.pyc
│     ├─ train.cpython-311.pyc
│     └─ __init__.cpython-311.pyc
├─ n.py
├─ nn
│  ├─ ffnn.py
│  ├─ __init__.py
│  └─ __pycache__
│     ├─ ffnn.cpython-311.pyc
│     └─ __init__.cpython-311.pyc
├─ pipe
│  ├─ pipeline.py
│  ├─ __init__.py
│  └─ __pycache__
│     ├─ pipeline.cpython-311.pyc
│     └─ __init__.cpython-311.pyc
├─ pipe.py
├─ README.md
├─ requirements.txt
├─ test.py
├─ tests
│  ├─ api_tests.py
│  ├─ __init__.py
│  └─ __pycache__
│     ├─ api_tests.cpython-311.pyc
│     └─ __init__.cpython-311.pyc
├─ utils
│  ├─ dataset_preprocessor.py
│  ├─ get_data_loader.py
│  ├─ k_fold.py
│  ├─ risk_score.py
│  ├─ split_data.py
│  ├─ __init__.py
│  └─ __pycache__
│     ├─ dataset_preprocessor.cpython-311.pyc
│     ├─ get_data_loader.cpython-311.pyc
│     ├─ k_fold.cpython-311.pyc
│     ├─ load_dataset.cpython-311.pyc
│     ├─ risk_score.cpython-311.pyc
│     ├─ split_data.cpython-311.pyc
│     ├─ train.cpython-311.pyc
│     └─ __init__.cpython-311.pyc
├─ web
│  ├─ api
│  │  ├─ server.py
│  │  ├─ __init__.py
│  │  └─ __pycache__
│  │     ├─ server.cpython-311.pyc
│  │     └─ __init__.cpython-311.pyc
│  ├─ public
│  │  └─ index.html
│  ├─ static
│  ├─ __init__.py
│  └─ __pycache__
│     └─ __init__.cpython-311.pyc
└─ __init__.py

```

---

## Installation

### Clone the repository
```bash
git clone https://github.com/DeveloperAromal/Fraud_Detection_FFNN.git
 cd Fraud_Detection_FFNN
```

----

