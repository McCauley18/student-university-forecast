import pickle
import shap
from pathlib import Path

MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "xgb_student_model_proper.pkl"



with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 

    
explainer = shap.TreeExplainer(model)
