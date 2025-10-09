import streamlit as st
from utils import predict_risk
from components.input_fields import get_input_features
from components.page_layout import set_page_layout

# -------------------- Layout -------------------- #
set_page_layout()

st.title("🎓 Student Academic Risk Predictor")
st.markdown("Use the sidebar to input student details and predict their academic risk level.")

# -------------------- Sidebar Inputs -------------------- #
input_features = get_input_features()

# -------------------- Predict Button -------------------- #
if st.button("🔮 Predict Academic Risk"):
    result = predict_risk(input_features)

    if "error" in result:
        st.error(f"⚠️ Error: {result['error']}")
    else:
        risk_labels = {0: "🚨 Dropout", 1: "📘 Enrolled", 2: "🎓 Graduate"}
        level = result["predicted_risk_level"]
        label = risk_labels.get(level, "Unknown")

        st.markdown(
            f"""
            <div style="
                background-color:#f0f2f6;
                border-radius:10px;
                padding:20px;
                text-align:center;
                font-size:22px;
                color:#333;
                box-shadow:0 4px 8px rgba(0,0,0,0.1);
            ">
                <b>Predicted Academic Risk Level:</b> {label}
            </div>
            """,
            unsafe_allow_html=True
        )
