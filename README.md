# Vietnamese Text Classification with TF-IDF and SVM

This project implements a complete machine learning pipeline for classifying Vietnamese text documents into predefined topics. It features data loading with robust encoding detection, dataset grouping, k-fold cross-validation, and model saving/loading for inference.

The core classifier uses a **TF-IDF Vectorizer** for text representation paired with a **Linear Support Vector Classifier (LinearSVC)**.

## 🚀 Features
- **Robust Text Reader**: Reads files using fallback encodings (`utf-8`, `utf-16`, `cp1258`, `cp1252`, `latin1`) to handle varying document formats.
- **Dynamic Grouping**: Automatically maps raw folder-based labels (e.g., "Soccer", "Basketball") to high-level categories (e.g., "Sports").
- **SVM Pipeline**: Implements a Scikit-Learn `Pipeline` combining `TfidfVectorizer` (word and bigram n-grams, sublinear TF scaling) and `LinearSVC`.
- **Comprehensive Evaluation**:
  - **Stratified K-Fold Cross-Validation** (5 splits) on the training set.
  - **Validation & Test Splits**: Evaluates performance on distinct validation (10%) and test (20%) datasets.
  - Outputs detailed metrics including Accuracy, Classification Reports (Precision, Recall, F1-Score), and Confusion Matrices.
- **Interactive Inference**: Saves the final trained model (`svm_vietnamese_text_classifier.joblib`) and supports interactive command-line text prediction.

---

## 📂 Project Structure
```text
Vietnamese-text-classification/
├── main.py                    # Main script (dataset loading, training, evaluation, and inference)
├── README.md                  # Project documentation (this file)
├── text.txt                   # Scratch text file
└── .gitignore                 # Git ignore configuration
```

---

## 🛠️ Requirements & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed. The project relies on the following Python packages:
- `numpy`
- `scikit-learn`
- `joblib`

### 2. Installation
Install the required packages using `pip`:
```bash
pip install numpy scikit-learn joblib
```

---

## 📈 Methodology & Workflow

### 1. Data Preparation
1. The script reads files from a configured root directory `DATA_DIR`.
2. Directory names represent the document classes. You can configure `group_labels` inside `main.py` to merge specific folders:
   - `Soccer`, `Basketball` ➡️ `Sports`
   - `Movies`, `Music` ➡️ `Entertainment`
   - `Tech` ➡️ `Technology`
   - `Politics` ➡️ `Politics`

### 2. Training Pipeline
- **TF-IDF Vectorizer**: Converts raw text to TF-IDF features.
  - `ngram_range=(1, 2)`: Captures both unigrams and bigrams.
  - `sublinear_tf=True`: Applies sublinear scaling to term frequencies ($1 + \log(\text{tf})$).
- **Linear SVC**: A Support Vector Machine classifier with a linear kernel, optimized for high-dimensional text data.
- **Data Splitting**: Splits the data into Train (70%), Validation (10%), and Test (20%).
- **Cross-Validation**: 5-Fold Stratified Cross-Validation on the training set to prevent overfitting.

### 3. Model Saving
After validation and testing, the pipeline retrains the model on the combined training and validation data (80% of the total dataset) and saves it to `svm_vietnamese_text_classifier.joblib`.

---

## 🏃 Usage

### 1. Configuration
Open `main.py` and modify the dataset directory path `DATA_DIR` to point to your data directory:
```python
DATA_DIR = r"/path/to/your/vietnamese/text/dataset"
```

### 2. Train and Evaluate
To run the full cross-validation and evaluation pipeline:
```bash
python main.py
```
This will train the model, display performance metrics (Accuracy, F1-Score, Confusion Matrix), and save the trained model.

### 3. Run Inference
The script includes an interactive console tool. After training, you can input sentences directly to get instant category predictions.

---

## 📝 License
This project is developed for educational and research purposes in NLP and Machine Learning.