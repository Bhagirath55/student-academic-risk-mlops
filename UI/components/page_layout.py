import streamlit as st

def set_page_layout():
    st.set_page_config(
        page_title="Student Academic Risk Prediction",
        page_icon="🎓",
        layout="wide"
    )
    st.markdown(
        """
        <style>
        /* Center title and make it cleaner */
        .stApp {
            background-color: #0E1117;
        }
        h1, h2, h3 {
            text-align: center;
            color: #1a1a1a;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
