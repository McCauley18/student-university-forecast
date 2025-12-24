# inference.py
import pandas as pd
from src.preprocessing import transform_data
from src.model import model, explainer

def run_inference(raw_input_df):
    # preprocess (no encoders needed)
    X = transform_data(raw_input_df)  # just clean & pass numbers

    # predict
    pred = model.predict(X)[0]
    prob = model.predict_proba(X)[0][1]

    # SHAP
    shap_values = explainer.shap_values(X)[0]
    contributions = sorted(
        zip(X.columns, shap_values),
        key=lambda x: abs(x[1]),
        reverse=True
    )

    return pred, prob, contributions

