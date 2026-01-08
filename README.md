# Product Category Prediction

This project implements a machine learning pipeline for predicting product categories based on product titles.

The solution uses TF-IDF text vectorization combined with a Linear Support Vector Classifier (LinearSVC).

---

## Project Structure

product-category-prediction/
│
├── data/
│ └── products.csv
│
├── models/
│ └── final_product_category_pipeline.pkl
│
├── notebooks/
│ └── 01_product_category_prediction.ipynb
│
├── train_model.py
├── predict_category.py
└── README.md


---

## Requirements

- Python 3.9+
- Required libraries:
  - pandas
  - scikit-learn
  - joblib

Install dependencies with:

```bash
pip install pandas scikit-learn joblib

---
## Training the Model
If you want to retrain the model from scratch:
- python train_model.py
# This script:
loads the dataset from data/products.csv
trains a TF-IDF + LinearSVC pipeline
saves the trained model to models/final_product_category_pipeline.pkl

---
## Running Predictions
To run interactive predictions using the trained model:
- python predict_category.py
You will be prompted to enter a product title, for example:
iphone 11 64gb black apple
The script will output the predicted product category.
To exit the program, press Enter on an empty line.
