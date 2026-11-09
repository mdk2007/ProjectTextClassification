import os
import sys
import re
import pandas as pd
from sklearn.model_selection import train_test_split

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from src.preprocess import preprocess_text, explain_preprocessing_steps

# 1. Nạp dữ liệu (Load Dataset)
# Ưu tiên đọc từ thư mục local data/, nếu chưa có sẽ tự động tải từ KaggleHub
DATA_PATH = os.path.join(os.path.dirname(__file__), "data", "IMDB Dataset.csv")

if not os.path.exists(DATA_PATH):
    print("Dataset local chưa có, đang tải từ KaggleHub...")
    import kagglehub
    import shutil
    path = kagglehub.dataset_download("lakshmi25npathi/imdb-dataset-of-50k-movie-reviews")
    os.makedirs(os.path.dirname(DATA_PATH), exist_ok=True)
    shutil.copy(os.path.join(path, "IMDB Dataset.csv"), DATA_PATH)

print(f"Đọc dữ liệu từ: {DATA_PATH}")
df = pd.read_csv(DATA_PATH)
print(f"Tổng số bản ghi: {len(df):,}")
print("5 dòng đầu tiên:")
print(df.head())

# 2. Tiền xử lý dữ liệu (NLP Preprocessing)
# Bao gồm: Lowercase, xóa HTML tag, mở rộng viết tắt (not), lọc stopwords thông minh, Lemmatization
print("\n--- Minh họa bước tiền xử lý trên 1 review mẫu ---")
sample_rev = df['review'].iloc[0][:150] + "..."
demo_steps = explain_preprocessing_steps(sample_rev)
print("Gốc:", demo_steps["raw_text"])
print("Đã làm sạch:", demo_steps["cleaned_text"])
print("Tokens giữ lại:", demo_steps["retained_tokens"][:8])
print("Sau Lemmatization:", demo_steps["lemmatized_tokens"][:8])

# 3. Chuyển đổi nhãn (Labels) thành số: 1 (Positive), 0 (Negative)
df['label'] = df['sentiment'].map({'positive': 1, 'negative': 0})
print("\nPhân bố nhãn:")
print(df['sentiment'].value_counts())

# 4. Chia tập Train/Test (80% huấn luyện, 20% kiểm thử)
X_train, X_test, y_train, y_test = train_test_split(
    df['review'], df['label'], test_size=0.2, random_state=42, stratify=df['label']
)
print(f"\nChia tập thành công: Train={len(X_train):,} reviews, Test={len(X_test):,} reviews")