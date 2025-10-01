import joblib
from preprocessing import load_and_clean
from train_model import train_models
from evaluate import evaluate_model

def save_best_model():
    # โหลดและ clean data
    X, y = load_and_clean("data/dataset_student.csv")
    models, X_train, X_test, y_train, y_test = train_models(X, y)

    # เลือกโมเดลที่ดีที่สุด
    best_model_name, results = evaluate_model(models, X_test, y_test)
    best_model = models[best_model_name]

    # เซฟโมเดล
    joblib.dump(best_model, "models/best_model.pkl")
    print(f"✅ Best model '{best_model_name}' saved to models/best_model.pkl")

if __name__ == "__main__":
    save_best_model()
