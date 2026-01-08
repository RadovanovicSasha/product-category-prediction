# train_model.py
# 9.1 — Treniranje i čuvanje finalnog modela (TF-IDF + LinearSVC)

import os
import pandas as pd
import joblib

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC


# Podesi putanje (ako ti je fajl drugačije lociran, promeni DATA_PATH)
DATA_PATH = "data/products.csv"
MODEL_DIR = "models"
MODEL_PATH = os.path.join(MODEL_DIR, "product_category_model.pkl")

# Kolone koje koristimo (po tvom df.columns)
TEXT_COL = "Product Title"
TARGET_COL = "Category Label"


def main():
    # 1) Učitaj podatke
    df = pd.read_csv(DATA_PATH)

    # 2) Minimalna validacija kolona
    missing = [c for c in [TEXT_COL, TARGET_COL] if c not in df.columns]
    if missing:
        raise ValueError(
            f"Nedostaju kolone: {missing}. "
            f"Kolone u fajlu su: {list(df.columns)}"
        )

    # 3) Minimalno čišćenje (samo da trening ne pukne)
    df = df.dropna(subset=[TEXT_COL, TARGET_COL])

    # 4) X i y
    X = df[TEXT_COL].astype(str)
    y = df[TARGET_COL].astype(str)

    # 5) Pipeline: TF-IDF + LinearSVC
    # Napomena: stop_words=None jer dataset ima mešavinu jezika/šifara/modela
    pipeline = Pipeline(
        steps=[
            ("tfidf", TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                max_features=20000
            )),
            ("clf", LinearSVC())
        ]
    )

    # 6) Treniranje na kompletnom skupu
    pipeline.fit(X, y)

    # 7) Čuvanje modela
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    # 8) Info log
    print("Finalni model istreniran na kompletnom skupu.")
    print(f"Tip finalnog modela: {type(pipeline).__name__}")
    print(f"Model uspešno sačuvan na lokaciji: {MODEL_PATH}")


if __name__ == "__main__":
    main()
