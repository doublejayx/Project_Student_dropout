import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix

# โหลด dataset
df = pd.read_csv("data/dataset_student.csv")

df = df.drop(columns=["student_id"], errors="ignore")
df = df.fillna(df.mean(numeric_only=True))
df = pd.get_dummies(df, drop_first=True)

y = df["Target"]

# แปลงเป็น binary classification: 1 = Dropout, 0 = อื่นๆ
y = y.apply(lambda x: 1 if x == "Dropout" else 0)

# ลบคอลัมน์ Target ออกจาก X
X = df.drop("Target", axis=1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)

print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))