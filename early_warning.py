import pandas as pd

def warning_system(model, X_test, y_test):
    proba = model.predict_proba(X_test)[:,1]

    warnings = [
        "High Risk" if p > 0.7 else "Medium Risk" if p > 0.4 else "Low Risk"
        for p in proba
    ]

    result = X_test.copy()
    result["True Label"] = y_test.values
    result["Predicted Risk"] = warnings
    result["Probability"] = proba

    print("\n=== Early Warning Results (10 ตัวอย่างแรก) ===")
    print(result.head(10))

    return result
