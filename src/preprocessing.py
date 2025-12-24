import pickle
import pandas as pd
from pathlib import Path

SCALER_PATH = Path(__file__).resolve().parent.parent / "model" / "scalerxgboost.pkl"

# Load scaler once

with open(SCALER_PATH, "rb") as f:
    scaler = pickle.load(f)

def transform_data(df):
    df_transformed = df.copy()
    
    # Drop missing
    df_transformed = df_transformed.dropna()
    
    # Map categorical columns
    type_school_map = {"Academic": 0, "Vocational": 1}
    school_accreditation_map = {"A": 0, "B": 1}
    gender_map = {"Male": 0, "Female": 1}
    interest_map = {"Less Interested": 0, "Uncertain": 1, "Interested": 2, "Very Interested": 3, "Not Interested": 4}
    residence_map = {"Urban": 0, "Rural": 1}

    df_transformed["type_school"] = df_transformed["type_school"].map(type_school_map)
    df_transformed["school_accreditation"] = df_transformed["school_accreditation"].map(school_accreditation_map)
    df_transformed["gender"] = df_transformed["gender"].map(gender_map)
    df_transformed["interest"] = df_transformed["interest"].map(interest_map)
    df_transformed["residence"] = df_transformed["residence"].map(residence_map)

    # Boolean -> int
    df_transformed['parent_was_in_college'] = df_transformed['parent_was_in_college'].astype(int)

    # Scale numeric columns
    numerical_cols = ['parent_age', 'parent_salary', 'house_area', 'average_grades']
    df_transformed[numerical_cols] = scaler.transform(df_transformed[numerical_cols])
    
    return df_transformed
