import joblib

MODEL_PATH = "product-category-prediction/models/final_product_category_pipeline.pkl"


def main():
    # Učitaj model (Pipeline)
    model = joblib.load(MODEL_PATH)
    print(f"Model uspešno učitan: {type(model).__name__}")
    print("Unesi naziv proizvoda (Enter za izlaz):\n")

    while True:
        text = input("> ").strip()
        if text == "":
            print("👋 Kraj.")
            break

        pred = model.predict([text])[0]
        print(f"Predikcija: {pred}\n")

if __name__ == "__main__":
    main()
