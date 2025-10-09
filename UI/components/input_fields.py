import streamlit as st

def get_input_features():
    st.sidebar.header("📋 Enter Student Information")

    # ---------------------- Student Info ---------------------- #
    with st.sidebar.expander("🧑 Student Details"):
        app_mode = st.number_input("Application Mode", min_value=0, step=1)
        app_order = st.number_input("Application Order", min_value=0, step=1)
        course = st.number_input("Course", min_value=0, step=1)
        attendance = st.selectbox("Daytime/Evening Attendance", [0, 1], help="0 = Evening, 1 = Daytime")
        prev_grade = st.number_input("Previous Qualification (Grade)", min_value=0.0, step=0.1)
        admission_grade = st.number_input("Admission Grade", min_value=0, step=1)
        displaced = st.selectbox("Displaced", [0, 1])
        debtor = st.selectbox("Debtor", [0, 1])
        tuition = st.selectbox("Tuition Fees Up To Date", [0, 1])
        gender = st.selectbox("Gender", [0, 1], help="0 = Female, 1 = Male")
        scholar = st.selectbox("Scholarship Holder", [0, 1])
        age = st.number_input("Age At Enrollment", min_value=15, max_value=30, step=1)

    # ---------------------- Semester 1 ---------------------- #
    with st.sidebar.expander("📘 Semester 1 Performance"):
        sem1_cred = st.number_input("1st Sem Units Credited", min_value=0, step=1)
        sem1_enroll = st.number_input("1st Sem Units Enrolled", min_value=0, step=1)
        sem1_eval = st.number_input("1st Sem Units Evaluations", min_value=0, step=1)
        sem1_approved = st.number_input("1st Sem Units Approved", min_value=0, step=1)
        sem1_grade = st.number_input("1st Sem Units Grade", min_value=0.0, step=0.1)

    # ---------------------- Semester 2 ---------------------- #
    with st.sidebar.expander("📗 Semester 2 Performance"):
        sem2_cred = st.number_input("2nd Sem Units Credited", min_value=0, step=1)
        sem2_enroll = st.number_input("2nd Sem Units Enrolled", min_value=0, step=1)
        sem2_eval = st.number_input("2nd Sem Units Evaluations", min_value=0, step=1)
        sem2_approved = st.number_input("2nd Sem Units Approved", min_value=0, step=1)
        sem2_grade = st.number_input("2nd Sem Units Grade", min_value=0.0, step=0.1)

    # ---------------------- Economic Indicators ---------------------- #
    with st.sidebar.expander("💹 Economic Indicators"):
        unemployment = st.number_input("Unemployment Rate", min_value=0.0, step=0.1)
        gdp = st.number_input("GDP", min_value=0.0, step=0.01)

    # Final dictionary (same order as your original code)
    features = {
        "Application mode": app_mode,
        "Application order": app_order,
        "Course": course,
        "Daytime/evening attendance": attendance,
        "Previous qualification (grade)": prev_grade,
        "Admission grade": admission_grade,
        "Displaced": displaced,
        "Debtor": debtor,
        "Tuition fees up to date": tuition,
        "Gender": gender,
        "Scholarship holder": scholar,
        "Age at enrollment": age,
        "Curricular units 1st sem (credited)": sem1_cred,
        "Curricular units 1st sem (enrolled)": sem1_enroll,
        "Curricular units 1st sem (evaluations)": sem1_eval,
        "Curricular units 1st sem (approved)": sem1_approved,
        "Curricular units 1st sem (grade)": sem1_grade,
        "Curricular units 2nd sem (credited)": sem2_cred,
        "Curricular units 2nd sem (enrolled)": sem2_enroll,
        "Curricular units 2nd sem (evaluations)": sem2_eval,
        "Curricular units 2nd sem (approved)": sem2_approved,
        "Curricular units 2nd sem (grade)": sem2_grade,
        "Unemployment rate": unemployment,
        "GDP": gdp
    }

    return features
