import pickle
import shap
from pathlib import Path

# MODEL_PATH = "../model/xgb_student_model_proper.pkl"

BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
MODEL_PATH = BASE_DIR / "model" / "xgb_student_model_proper.pkl"


with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)
 

    
explainer = shap.TreeExplainer(model)
