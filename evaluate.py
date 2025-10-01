from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

def evaluate_model(models, X_test, y_test):
    results = {}

    for name, model in models.items():
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred)
        rec = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        results[name] = {
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1
        }

        print(f"\n{name}:")
        print(confusion_matrix(y_test, y_pred))
        print(classification_report(y_test, y_pred))

    # หาว่าโมเดลไหนดีที่สุด (ใช้ f1 เป็นหลัก)
    best_model = max(results, key=lambda k: results[k]["f1"])
    print("\n==============================")
    print(f"⭐ Best Model: {best_model}")
    print("Performance:", results[best_model])
    print("==============================")

    return best_model, results
