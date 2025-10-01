# 🎓 Student Dropout Early Warning System

ระบบ Machine Learning เพื่อทำนายความเสี่ยงนักเรียนหลุดจากระบบการศึกษา (Dropout)  
และแสดงผลผ่าน **Streamlit Dashboard** แบบเรียลไทม์ 🚀  

---

## 📂 โครงสร้างโปรเจกต์
student_dropout_project/
├── data/
│ └── dataset_student.csv # ข้อมูลนักเรียน
├── models/
│ └── best_model.pkl # โมเดลที่เทรนและบันทึกแล้ว
├── preprocessing.py # เตรียมและเลือก features
├── train_model.py # เทรนโมเดล
├── evaluate.py # ประเมินผลโมเดล
├── early_warning.py # ระบบแจ้งเตือนนักเรียนเสี่ยง
├── main.py # pipeline หลัก
├── save_model.py # บันทึกโมเดลที่ดีที่สุด
├── dashboard.py # Streamlit Dashboard
