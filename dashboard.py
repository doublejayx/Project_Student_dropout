import streamlit as st
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix
from preprocessing import load_and_clean

# --- Page Configuration ---
st.set_page_config(
    page_title="Student Dropout Dashboard",
    page_icon="🎓",
    layout="wide"
)

# --- Custom CSS for Styling ---
def apply_custom_css():
    css = """
    /* --- General Styles --- */
    body {
        background-color: #f0f2f6;
    }
    .stApp {
        background-color: #f0f2f6;
    }
    
    /* --- Title --- */
    h1 {
        font-size: 2.8rem !important;
        font-weight: 700 !important;
        color: #1E3A8A; /* Dark Blue */
    }

    /* --- Subheader --- */
    h2 {
        margin-bottom: 0.5rem !important; /* Reduces space below the subheader */
    }

    /* --- Metric Cards --- */
    div[data-testid="stMetric"] {
        background-color: #FFFFFF;
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        transition: transform 0.2s ease-in-out, box-shadow 0.2s ease-in-out;
    }
    div[data-testid="stMetric"]:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 24px rgba(0,0,0,0.1);
    }
    div[data-testid="stMetric"] > label[data-testid="stMetricLabel"] {
        font-size: 1.1rem !important;
        font-weight: 500 !important;
        color: #4A5568; /* Gray */
    }
    div[data-testid="stMetric"] > div[data-testid="stMetricValue"] {
        font-size: 2.5rem !important;
        font-weight: 700 !important;
    }
    
    /* --- Sidebar --- */
    [data-testid="stSidebar"] {
        background-color: #FFFFFF;
        border-right: 1px solid #E0E0E0;
    }
    
    /* --- Expander --- */
    [data-testid="stExpander"] {
        border: 1px solid #E0E0E0;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
    }
    
    /* --- Chart container --- */
    .chart-container {
        background-color: #FFFFFF;
        border-radius: 12px;
        padding: 20px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.05);
        height: 100%;
    }
    """
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)

apply_custom_css()


# --- Load Model and Data ---
@st.cache_data
def load_model_and_data():
    """Loads the pre-trained model and student data."""
    try:
        model = joblib.load("models/best_model.pkl")
    except FileNotFoundError:
        st.error("Model file 'models/best_model.pkl' not found. Please ensure the model exists.")
        st.stop()
    X, y = load_and_clean("data/dataset_student.csv")
    return model, X, y

model, X, y = load_model_and_data()

# --- Prediction and DataFrame Creation ---
@st.cache_data
def make_predictions(_model, X, y):
    """Generates predictions and creates the result DataFrame."""
    proba = _model.predict_proba(X)[:, 1]
    y_pred_result = _model.predict(X)
    
    def assign_risk(p):
        if p > 0.7: return "High Risk"
        if p > 0.4: return "Medium Risk"
        return "Low Risk"
            
    warnings = [assign_risk(p) for p in proba]

    df_result = X.copy()
    df_result["True Label"] = y.values
    df_result["Probability"] = proba
    df_result["Risk Level"] = warnings
    return df_result, y_pred_result

df_result, y_pred = make_predictions(model, X, y)

# --- Model Performance Metrics Calculation ---
cm = confusion_matrix(y, y_pred)
tn, fp, fn, tp = cm.ravel()

accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
precision = tp / (tp + fp) if (tp + fp) > 0 else 0
recall = tp / (tp + fn) if (tp + fn) > 0 else 0
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0


# --- UI Customization ---
RISK_COLORS = {
    "High Risk": "#FF4B4B",
    "Medium Risk": "#FFC107",
    "Low Risk": "#28A745"
}

# --- Sidebar ---
with st.sidebar:
    st.image("photo/student2.png", width='stretch')
    st.title("⚙️ Control Panel")
    st.markdown("เลือกกลุ่มความเสี่ยงเพื่อกรองข้อมูลและไฮไลท์กราฟ")
    risk_filter = st.selectbox(
        "เลือกกลุ่มความเสี่ยง",
        ["All", "High Risk", "Medium Risk", "Low Risk"],
        label_visibility="collapsed"
    )
    st.info("แดชบอร์ดนี้เป็นเครื่องมือช่วยสนับสนุนการตัดสินใจ ไม่สามารถใช้แทนการประเมินรายบุคคลได้")


# --- Main Dashboard ---
st.title("🎓 Student Dropout Early Warning")
st.markdown("##### ระบบแดชบอร์ดสำหรับเฝ้าระวังและวิเคราะห์ความเสี่ยงที่นักเรียนจะออกกลางคัน (Dropout)")
st.markdown("""
- 🔴 **High Risk**: มีความเสี่ยงสูงที่จะออกกลางคัน ควรได้รับการดูแลอย่างใกล้ชิด
- 🟡 **Medium Risk**: มีความเสี่ยงปานกลาง อาจต้องการคำปรึกษาเพิ่มเติม
- 🟢 **Low Risk**: มีความเสี่ยงต่ำ แต่ยังคงต้องเฝ้าระวัง
""")
st.divider()

# --- Model Performance KPIs ---
st.subheader("⚙️ ภาพรวมประสิทธิภาพโมเดล")
kpi_cols = st.columns(4)
with kpi_cols[0]:
    st.metric(label="Accuracy", value=f"{accuracy:.2%}")
with kpi_cols[1]:
    st.metric(label="Precision", value=f"{precision:.2%}")
with kpi_cols[2]:
    st.metric(label="Recall", value=f"{recall:.2%}")
with kpi_cols[3]:
    st.metric(label="F1-Score", value=f"{f1_score:.2%}")
st.divider()


# --- Filtered DataFrame ---
filtered_df = df_result[df_result["Risk Level"] == risk_filter] if risk_filter != "All" else df_result

# --- Summary Metrics ---
st.subheader("📊 สรุปภาพรวมความเสี่ยง")
risk_counts = df_result["Risk Level"].value_counts()
total_students = len(df_result)
high_risk_count = risk_counts.get('High Risk', 0)
medium_risk_count = risk_counts.get('Medium Risk', 0)
low_risk_count = risk_counts.get('Low Risk', 0)

cols = st.columns(4)
with cols[0]:
    st.metric(label="👥 นักเรียนทั้งหมด", value=f"{total_students:,}")
with cols[1]:
    st.metric(label="🔴 High Risk", value=f"{high_risk_count:,}")
with cols[2]:
    st.metric(label="🟡 Medium Risk", value=f"{medium_risk_count:,}")
with cols[3]:
    st.metric(label="🟢 Low Risk", value=f"{low_risk_count:,}")

st.divider()

# --- Charts Section ---
st.subheader("📈 การกระจายตัวและความเสี่ยง")
col1, col2 = st.columns([2, 3])
labels = ["High Risk", "Medium Risk", "Low Risk"]

with col1:
    with st.container(border=False):
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<h6>🥧 สัดส่วนกลุ่มเสี่ยง</h6>", unsafe_allow_html=True)
        sizes = [risk_counts.get(label, 0) for label in labels]
        pull = [0.1 if label == risk_filter else 0 for label in labels]

        fig_pie = go.Figure(data=[go.Pie(
            labels=labels, values=sizes, hole=.4, pull=pull,
            marker_colors=[RISK_COLORS[label] for label in labels]
        )])
        fig_pie.update_layout(
            legend=dict(orientation="h", yanchor="bottom", y=-0.2, xanchor="center", x=0.5),
            margin=dict(l=20, r=20, t=20, b=20), showlegend=True
        )
        st.plotly_chart(fig_pie, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col2:
    with st.container(border=False):
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<h6>📶 จำนวนนักเรียนในแต่ละกลุ่ม</h6>", unsafe_allow_html=True)
        chart_data = risk_counts.reset_index()
        chart_data.columns = ['Risk Level', 'Number of Students']

        fig_bar = px.bar(
            chart_data, x='Risk Level', y='Number of Students', color='Risk Level',
            color_discrete_map=RISK_COLORS, category_orders={"Risk Level": labels}
        )
        
        for trace in fig_bar.data:
            if risk_filter != "All" and trace.name != risk_filter:
                trace.opacity = 0.5
            else:
                trace.opacity = 1.0

        fig_bar.update_layout(showlegend=False, xaxis_title=None, yaxis_title="Number of Students")
        st.plotly_chart(fig_bar, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

# --- Model Deep Dive Analysis ---
st.divider()
st.subheader("🔬 วิเคราะห์โมเดลเชิงลึก (Model Deep Dive)")
col3, col4 = st.columns(2)

with col3:
    with st.container(border=False):
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown(f"<h6>📈 การกระจายตัวความน่าจะเป็น ({risk_filter})</h6>", unsafe_allow_html=True)
        
        fig_hist = px.histogram(
            filtered_df, x="Probability", nbins=30,
            labels={"Probability": "Predicted Probability"},
            color_discrete_sequence=[RISK_COLORS.get(risk_filter, '#1E3A8A')] if risk_filter != "All" else ['#1E3A8A']
        )
        fig_hist.update_layout(yaxis_title="Number of Students", showlegend=False, title_text=None)
        st.plotly_chart(fig_hist, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

with col4:
    with st.container(border=False):
        st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
        st.markdown("<h6>🧮 Confusion Matrix (Overall)</h6>", unsafe_allow_html=True)
        cm_labels = ["Not Dropout", "Dropout"]
        fig_cm = go.Figure(data=go.Heatmap(
            z=cm, x=cm_labels, y=cm_labels, text=cm, texttemplate="%{text}", colorscale='Blues'
        ))
        fig_cm.update_layout(xaxis_title="Predicted Label", yaxis_title="True Label")
        st.plotly_chart(fig_cm, use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)


# --- Feature Importance Analysis ---
st.divider()
st.subheader("🌟 ปัจจัยสำคัญที่ส่งผลต่อความเสี่ยง (Key Risk Factors)")

with st.container():
    st.markdown("<div class='chart-container'>", unsafe_allow_html=True)
    st.markdown("กราฟนี้แสดงปัจจัย 15 อันดับแรกที่มีผลต่อการทำนายของโมเดลมากที่สุด ยิ่งค่า **Importance Score** สูง แสดงว่าปัจจัยนั้นยิ่งมีความสำคัญ")
    
    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        feat_importances = pd.Series(importances, index=X.columns)
        top_features = feat_importances.nlargest(15).sort_values(ascending=True)

        fig_feat = go.Figure()
        fig_feat.add_trace(go.Bar(
            x=top_features.values, y=top_features.index, orientation='h',
            marker=dict(color=top_features.values, colorscale="Blues_r", reversescale=True),
        ))
        fig_feat.update_layout(
            title="Top 15 Most Important Features",
            xaxis_title="Importance Score", yaxis_title=None,
            height=500
        )
        st.plotly_chart(fig_feat, use_container_width=True)
    else:
        st.info("⚠️ โมเดลนี้ไม่รองรับการแสดง Feature Importance (เช่น Logistic Regression, KNN)")
        
    st.markdown("</div>", unsafe_allow_html=True)

# --- Data Table and Download ---
st.divider()
with st.expander("📋 รายชื่อนักเรียนและความเสี่ยง (คลิกเพื่อแสดง/ซ่อน)", expanded=False):
    st.markdown(f"**ตารางแสดงผลสำหรับกลุ่ม: `{risk_filter}`** (แสดงผล 50 อันดับแรก)")
    st.dataframe(filtered_df.head(50), width='stretch')
    
    @st.cache_data
    def convert_df_to_csv(df):
        return df.to_csv(index=False).encode('utf-8')

    csv = convert_df_to_csv(df_result)

    st.download_button(
       label="📥 ดาวน์โหลดผลการวิเคราะห์ทั้งหมด (CSV)",
       data=csv,
       file_name="student_dropout_analysis.csv",
       mime="text/csv",
    )

