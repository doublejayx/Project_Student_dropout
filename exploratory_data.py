import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ==============================
# Load Dataset
# ==============================
df = pd.read_csv("data/dataset_student.csv")

# ==============================
# Create folder for photos
# ==============================
PHOTO_DIR = "photo"
os.makedirs(PHOTO_DIR, exist_ok=True)

# ==============================
# 1. ข้อมูลเบื้องต้น
# ==============================
print("🔎 Dataset Shape:", df.shape)
print("\n📋 Columns:", df.columns.tolist())
print("\n🧾 Data Types:\n", df.dtypes)
print("\n📊 Missing Values:\n", df.isnull().sum())

# ==============================
# 2. Target Distribution
# ==============================
plt.figure(figsize=(6,4))
sns.countplot(x="Target", hue="Target", data=df, palette="Set2", legend=False)
plt.title("Distribution of Target (Dropout / Graduate / Enrolled)")
plt.xlabel("Target")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PHOTO_DIR, "eda_target_distribution.png"))
plt.close()

# ==============================
# 3. Age Distribution
# ==============================
plt.figure(figsize=(6,4))
sns.histplot(df["Age at enrollment"], bins=20, kde=True, color="blue")
plt.title("Age Distribution of Students")
plt.xlabel("Age at enrollment")
plt.ylabel("Count")
plt.tight_layout()
plt.savefig(os.path.join(PHOTO_DIR, "eda_age_distribution.png"))
plt.close()

# ==============================
# 4. Boxplot: Grade vs Target
# ==============================
plt.figure(figsize=(8,5))
sns.boxplot(
    x="Target",
    y="Curricular units 1st sem (grade)",
    data=df,
    hue="Target",           # บอกให้ใช้สีตาม Target
    palette="Set3",
    legend=False            # ปิด legend เพราะซ้ำกับแกน X
)
plt.title("Boxplot: 1st Semester Grade vs Target")
plt.xlabel("Target")
plt.ylabel("Grade (1st Sem)")
plt.tight_layout()
plt.savefig(os.path.join(PHOTO_DIR, "eda_grade_vs_target.png"))
plt.close()

# ==============================
# 5. Correlation Heatmap (numeric features)
# ==============================
plt.figure(figsize=(12,8))
numeric_df = df.select_dtypes(include=["int64", "float64"])
corr = numeric_df.corr()
sns.heatmap(corr, cmap="coolwarm", annot=False)
plt.title("Correlation Heatmap (Numeric Features)")
plt.tight_layout()
plt.savefig(os.path.join(PHOTO_DIR, "eda_correlation_heatmap.png"))
plt.close()

print("✅ EDA เสร็จสิ้น → รูปทั้งหมดถูกเก็บไว้ในโฟลเดอร์ 'photo/' แล้ว")