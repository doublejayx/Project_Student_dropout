import pandas as pd
from sklearn.ensemble import RandomForestClassifier

def load_and_clean(path="data/dataset_student.csv", top_n_features=15):
    df = pd.read_csv(path)

    # จัดการ missing values
    df = df.fillna(df.mean(numeric_only=True))

    # Target: Dropout = 1, อื่นๆ = 0
    y = df["Target"].apply(lambda x: 1 if x == "Dropout" else 0)

    # ลบ Target ออกจาก X
    X = df.drop("Target", axis=1)

    # One-hot encoding เฉพาะ X
    X = pd.get_dummies(X, drop_first=True)

    # -----------------------------
    # เลือก Top-N Features โดยใช้ RandomForest
    # -----------------------------
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X, y)

    feat_importances = pd.Series(rf.feature_importances_, index=X.columns)
    top_features = feat_importances.nlargest(top_n_features).index.tolist()

    print(f"✅ Using Top {top_n_features} Features: {top_features}")

    return X[top_features], y
