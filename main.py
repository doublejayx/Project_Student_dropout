from preprocessing import load_and_clean
from train_model import train_models
from evaluate import evaluate_model
from early_warning import warning_system

# 1. โหลดและทำความสะอาดข้อมูล พร้อมเลือก Top Features
X, y = load_and_clean("data/dataset_student.csv", top_n_features=15)

# 2. เทรนโมเดล
models, X_train, X_test, y_train, y_test = train_models(X, y)

# 3. ประเมินผล + เลือกโมเดลที่ดีที่สุด
best_model_name, results = evaluate_model(models, X_test, y_test)
best_model = models[best_model_name]

# 4. ระบบแจ้งเตือนความเสี่ยง
result = warning_system(best_model, X_test, y_test)
