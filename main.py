import os
from typing import List, Tuple
from collections import Counter

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_val_predict, train_test_split
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import joblib

# ==========================
# CẤU HÌNH ĐƯỜNG DẪN DỮ LIỆU
# ==========================
DATA_DIR = r"D:\Natural language processing\data\data\new data"
MODEL_PATH = "svm_vietnamese_text_classifier.joblib"

N_SPLITS = 5
RANDOM_STATE = 42


# ==========================
# HÀM ĐỌC FILE TEXT
# ==========================
def read_text_file(path: str) -> str:
    encodings = ["utf-8", "utf-16", "cp1258", "cp1252", "latin1"]
    for enc in encodings:
        try:
            with open(path, "r", encoding=enc) as f:
                return f.read()
        except UnicodeDecodeError:
            continue

    # fallback: đọc binary rồi decode lỗi
    with open(path, "rb") as f:
        return f.read().decode("latin1", errors="ignore")


# ==========================
# LOAD DỮ LIỆU (DUYỆT TẤT CẢ THƯ MỤC CON VÀ GỘP CHỦ ĐỀ)
# ==========================
def load_dataset(root_dir: str, group_labels: dict = None) -> Tuple[List[str], List[str], List[str]]:
    texts: List[str] = []
    labels: List[str] = []
    paths: List[str] = []

    if not os.path.isdir(root_dir):
        raise ValueError(f"Thư mục không tồn tại: {root_dir}")

    print(f"Đang duyệt dữ liệu trong thư mục: {root_dir}")

    for dirpath, _, filenames in os.walk(root_dir):
        # bỏ qua thư mục gốc vì nó không phải nhãn
        if dirpath == root_dir:
            continue

        label = os.path.basename(dirpath)

        # Nếu có group_labels, gán lại nhãn theo nhóm
        if group_labels and label in group_labels:
            label = group_labels[label]  # Gán nhãn mới theo nhóm

        txt_count_in_dir = 0
        for fname in filenames:
            if not fname.lower().endswith(".txt"):
                continue

            fpath = os.path.join(dirpath, fname)

            try:
                text = read_text_file(fpath)
            except Exception as e:
                print(f"Lỗi khi đọc file {fpath}: {e}")
                continue

            if not text.strip():
                continue

            texts.append(text)
            labels.append(label)
            paths.append(fpath)
            txt_count_in_dir += 1

        if txt_count_in_dir > 0:
            print(f"- Thư mục: {dirpath} -> {txt_count_in_dir} file .txt, nhãn = '{label}'")

    print(f"Tổng số văn bản đọc được từ {root_dir}: {len(texts)}")
    return texts, labels, paths


# ==========================
# TẠO PIPELINE SVM + TF-IDF
# ==========================
def build_model() -> Pipeline:
    return Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1, 2),
            sublinear_tf=True,
        )),
        ("clf", LinearSVC())
    ])


# ==========================
# CHIA TẬP DỮ LIỆU TRAIN, VALIDATION, TEST
# ==========================
def split_dataset(X: List[str], y: List[str], test_size: float = 0.2, validation_size: float = 0.1,
                  random_state: int = 42):
    # Chia trước 80% dữ liệu cho train + validation
    X_train_val, X_test, y_train_val, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    # Từ tập train + validation, chia lại thành train + validation (70% train, 10% validation)
    validation_size_adjusted = validation_size / (
            1 - test_size)  # Điều chỉnh tỷ lệ cho phù hợp với tập con (80% còn lại)
    X_train, X_val, y_train, y_val = train_test_split(X_train_val, y_train_val, test_size=validation_size_adjusted,
                                                      random_state=random_state)

    print(f"Tập Train: {len(X_train)} samples")
    print(f"Tập Validation: {len(X_val)} samples")
    print(f"Tập Test: {len(X_test)} samples")

    return X_train, X_val, X_test, y_train, y_val, y_test


# ==========================
# CROSS VALIDATION + TRAIN FULL
# ==========================
def cross_validate_and_train_full():
    print("==== LOAD DỮ LIỆU ====")

    # Dữ liệu gộp chủ đề
    group_labels = {
        "Soccer": "Sports",
        "Basketball": "Sports",
        "Politics": "Politics",
        "Movies": "Entertainment",
        "Music": "Entertainment",
        "Tech": "Technology"
    }

    X, y, paths = load_dataset(DATA_DIR, group_labels=group_labels)

    if not X:
        raise RuntimeError("Không có dữ liệu. Kiểm tra lại DATA_DIR.")

    unique_labels = sorted(set(y))
    print(f"Số văn bản: {len(X)}")
    print(f"Tập nhãn: {unique_labels}")

    # Chia dữ liệu thành Train (70%), Validation (10%), và Test (20%)
    X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y)

    print("\n==== CROSS VALIDATION ====")
    model = build_model()

    # Cross-validation với tập train (70%)
    cv = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=RANDOM_STATE)
    scores = cross_val_score(model, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)
    print(f"Accuracy từng fold: {np.round(scores, 4)}")
    print(f"Accuracy trung bình: {scores.mean():.4f} (+/- {scores.std():.4f})")

    # Huấn luyện và đánh giá trên tập validation (10%)
    model.fit(X_train, y_train)
    y_val_pred = model.predict(X_val)
    print("\n==== ĐÁNH GIÁ TẬP VALIDATION ====")
    print(f"Accuracy (Validation): {accuracy_score(y_val, y_val_pred):.4f}\n")

    print("=== Classification report (Validation) ===")
    print(classification_report(y_val, y_val_pred))

    print("=== Confusion matrix (Validation) ===")
    print(confusion_matrix(y_val, y_val_pred, labels=unique_labels))

    # 3) Đánh giá mô hình trên tập test (20%)
    y_test_pred = model.predict(X_test)
    print("\n==== ĐÁNH GIÁ TẬP TEST ====")
    print(f"Accuracy (Test): {accuracy_score(y_test, y_test_pred):.4f}\n")

    print("=== Classification report (Test) ===")
    print(classification_report(y_test, y_test_pred))

    print("=== Confusion matrix (Test) ===")
    print(confusion_matrix(y_test, y_test_pred, labels=unique_labels))

    # 4) Lưu mô hình sau khi huấn luyện trên toàn bộ dữ liệu train
    print("\n==== TRAIN TRÊN TOÀN BỘ DATA & LƯU MODEL ====")
    model.fit(X_train + X_val, y_train + y_val)
    joblib.dump(model, MODEL_PATH)
    print(f"Đã lưu model vào: {MODEL_PATH}")


# ==========================
# DỰ ĐOÁN 1 VĂN BẢN
# ==========================
def predict_one(text: str, model_path: str = MODEL_PATH) -> str:
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Không tìm thấy model: {model_path}. Hãy chạy cross_validate_and_train_full() trước."
        )

    model: Pipeline = joblib.load(model_path)
    return model.predict([text])[0]


# ==========================
# DỰ ĐOÁN 1 VĂN BẢN NHẬP TỪ BÀN PHÍM
# ==========================
def predict_from_input(model_path: str = MODEL_PATH) -> str:
    # Lấy dữ liệu từ bàn phím
    sample_text = input("Nhập văn bản cần dự đoán chủ đề: ")

    # Dự đoán chủ đề cho văn bản nhập vào
    print("Dự đoán chủ đề:", predict_one(sample_text, model_path))


# ==========================
# MAIN
# ==========================
if __name__ == "__main__":
    cross_validate_and_train_full()

    # Thêm phần nhập dữ liệu từ bàn phím
    predict_from_input()
