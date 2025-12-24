import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
import warnings
warnings.filterwarnings('ignore')
from src.inference import run_inference
#from src.preprocessing import encoders  # loaded from training

# Set page configuration
st.set_page_config(
    page_title="University Enrollment Predictor",
    layout="wide"
)

# Custom CSS for better styling
st.markdown("""
<style>
    
    /* Set entire page background color */
    .stApp {
        background-color: #0F172A;
    }
            
    .main-header {
        font-size: 2.5rem;
        color: #3498DB;
        text-align: center;
        margin-bottom: 2rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #34495e;
        margin-top: 2rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #f8f9fa;
        padding: 2rem;
        border-radius: 10px;
        border-left: 5px solid #3498db;
        margin-top: 2rem;
    }
    .high-prob {
        color: #27ae60;
        font-weight: bold;
    }
    .medium-prob {
        color: #f39c12;
        font-weight: bold;
    }
    .low-prob {
        color: #e74c3c;
        font-weight: bold;
    }
    .stButton button {
        width: 100%;
        background-color: #3498db;
        color: white;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        border: none;
    }
    .stButton button:hover {
        background-color: #2980b9;
    }
</style>
""", unsafe_allow_html=True)

# App title
st.markdown('<h1 class="main-header">University Enrollment Predictor</h1>', unsafe_allow_html=True)

# Create two columns for input
col1, col2 = st.columns(2)

with col1:
    st.markdown('<h3 class="sub-header">Student Information</h3>', unsafe_allow_html=True)
    
    # School Type
    type_school = st.selectbox(
        "School Type",
        ["Academic", "Vocational"],
        help="Type of school the student attends"
    )
    
    # School Accreditation
    school_accreditation = st.selectbox(
        "School Accreditation",
        ["A", "B"],
        help="Accreditation level of the school"
    )
    
    # Gender
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"],
        help="Student's gender"
    )
    
    # Interest Level
    interest = st.selectbox(
        "Interest in University",
        ["Less Interested", "Very Interested", "Uncertain" , "Not Interested" , "Interested"],
        help="Student's interest in attending university"
    )
    
    # Residence
    residence = st.selectbox(
        "Residence Area",
        ["Urban", "Rural"],
        help="Area where student resides"
    )

with col2:
    st.markdown('<h3 class="sub-header">Parent & Academic Information</h3>', unsafe_allow_html=True)
    
    # Parent Age
    parent_age = st.slider(
        "Parent Age",
        min_value=30,
        max_value=70,
        value=50,
        help="Age of the parent/guardian"
    )
    
    # Parent Salary
    parent_salary = st.number_input(
        "Parent Salary (in millions)",
        min_value=1000000,
        max_value=20000000,
        value=5000000,
        step=100000,
        help="Annual salary of parent in local currency"
    )
    
    # House Area
    house_area = st.slider(
        "House Area (sqm)",
        min_value=30.0,
        max_value=200.0,
        value=80.0,
        step=0.1,
        help="Area of the house in square meters"
    )
    
    # Average Grades
    average_grades = st.slider(
        "Average Grades (%)",
        min_value=0.0,
        max_value=100.0,
        value=85.0,
        step=0.1,
        help="Student's average academic grades"
    )
    
    # Parent was in university
    parent_was_in_college = st.selectbox(
        "Parent College Education",
        ["Yes", "No"],
        help="Did the parent attend university?"
    )

# Convert categorical inputs to appropriate format
parent_was_in_college_bool = parent_was_in_college == "Yes"

# Display current inputs in a summary
st.markdown('<h3 class="sub-header">Input Summary</h3>', unsafe_allow_html=True)

summary_cols = st.columns(5)
with summary_cols[0]:
    st.metric("School Type", type_school)
with summary_cols[1]:
    st.metric("Accreditation", school_accreditation)
with summary_cols[2]:  
    st.metric("Gender", gender)
with summary_cols[3]:
    st.metric("Interest", interest)
with summary_cols[4]:
    st.metric("Residence", residence)

summary_cols2 = st.columns(5)
with summary_cols2[0]:
    st.metric("Parent Age", parent_age)
with summary_cols2[1]:
    st.metric("Parent Salary", f"{parent_salary/1000000:.1f}M")
with summary_cols2[2]:
    st.metric("House Area", f"{house_area} sqm")
with summary_cols2[3]:
    st.metric("Avg Grades", f"{average_grades}%")
with summary_cols2[4]:
    st.metric("Parent University", parent_was_in_college)

# Prediction button
st.markdown("---")
predict_button = st.button("Predict University Enrollment Probability")

# Sample model function (you would replace this with your actual trained model)
# def predict_college_enrollment(input_features):
   
    
#     features = pd.DataFrame([input_features])
    
     
#     probability = 0.5
    
    
#     if input_features['interest'] == "Very Interested":
#         probability += 0.2
#     if input_features['average_grades'] > 85:
#         probability += 0.15
#     if input_features['parent_was_in_college']:
#         probability += 0.1
#     if input_features['school_accreditation'] == "A":
#         probability += 0.05
#     if input_features['parent_salary'] > 5000000:
#         probability += 0.05
#     if input_features['type_school'] == "Academic":
#         probability += 0.05
    
     
#     probability = max(0.1, min(0.95, probability))
    
#     return probability

if predict_button:
    # Build input dataframe from user inputs
    input_df = pd.DataFrame([{
        'type_school': type_school,
        'school_accreditation': school_accreditation,
        'gender': gender,
        'interest': interest,
        'residence': residence,
        'parent_age': parent_age,
        'parent_salary': parent_salary,
        'house_area': house_area,
        'average_grades': average_grades,
        'parent_was_in_college': parent_was_in_college_bool
    }])

    # Convert categorical to numeric using fixed mappings
    type_school_map = {"Academic": 0, "Vocational": 1}
    school_accreditation_map = {"A": 0, "B": 1}
    gender_map = {"Male": 0, "Female": 1}
    interest_map = {"Less Interested": 0, "Uncertain": 1, "Interested": 2, "Very Interested": 3, "Not Interested": 4}
    residence_map = {"Urban": 0, "Rural": 1}

    input_df["type_school"] = input_df["type_school"].map(type_school_map)
    input_df["school_accreditation"] = input_df["school_accreditation"].map(school_accreditation_map)
    input_df["gender"] = input_df["gender"].map(gender_map)
    input_df["interest"] = input_df["interest"].map(interest_map)
    input_df["residence"] = input_df["residence"].map(residence_map)

    # Run inference
    pred, prob, shap_contribs = run_inference(input_df)

    # Display SHAP explanation
    FRIENDLY_NAMES = {
        "average_grades": "Academic performance",
        "interest": "Interest in university",
        "parent_was_in_college": "Parental education",
        "parent_salary": "Financial background",
        "type_school": "School type",
        "residence": "Living area"
    }

    st.markdown("### 🔍 Why this prediction was made")
    for feature, value in shap_contribs[:5]:
        name = FRIENDLY_NAMES.get(feature, feature)
        if value > 0:
            st.write(f"✅ {name} increased likelihood")
        else:
            st.write(f"❌ {name} reduced likelihood")

    # Final output
    if prob >= 0.7:
        verdict = "Very likely to attend university"
    elif prob >= 0.4:
        verdict = "May attend university"
    else:
        verdict = "Unlikely to attend university"

    st.metric("Prediction", verdict)
    st.metric("Confidence", f"{prob*100:.1f}%")

    st.markdown('</div>', unsafe_allow_html=True)

# Sidebar with information
with st.sidebar:
    st.markdown("## ℹ️ About This App")
    st.markdown("""
    This app predicts the probability of a student enrolling in university based on various factors including:
    
    - **Student Demographics**: Gender, residence
    - **Academic Factors**: School type, grades, accreditation
    - **Family Background**: Parent education, salary, age
    - **Student Interest**: Level of interest in university
    
    ### How to use:
    1. Fill in all the student information
    2. Adjust sliders and select options
    3. Click the **Predict** button
    4. View the enrollment probability and recommendations
    """)
    
    st.markdown("---")
    st.markdown("### Sample Data Statistics")
    
    # Sample statistics (you can update these based on your data)
    stats_data = {
        "Metric": ["Avg Parent Age", "Avg Parent Salary", "Avg House Area", "Avg Grades"],
        "Value": ["48-55 years", "4-6 million", "75-85 sqm", "82-87%"]
    }
    
    stats_df = pd.DataFrame(stats_data)
    st.table(stats_df)

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d;">
    <p>University Enrollment Predictor | Made with Streamlit</p>
</div>
""", unsafe_allow_html=True)