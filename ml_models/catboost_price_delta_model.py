import os
import pandas as pd
from catboost import CatBoostRegressor

# Path to the .cbm file (adjust if needed)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "catboost_price_delta_model.cbm")

# Feature names in the order the model expects
FEATURES = [
    "location",
    "effective_demand",
    "stock_level",
    "competitor_price",
    "weather",
    "product_name",
    "category",
    "base_price",
    "shelf_life_days"
]

# Load the trained CatBoost model once on import
_model: CatBoostRegressor
try:
    _model = CatBoostRegressor()
    _model.load_model(MODEL_PATH)
except Exception as e:
    raise RuntimeError(f"Failed to load CatBoost model at {MODEL_PATH}: {e}")

def predict_price_delta(input_data):
    """
    Predicts the price delta percentage using the CatBoost model.
    
    Parameters
    ----------
    input_data : dict or list
        If dict, keys must match FEATURES.  
        If list, values must be in the same order as FEATURES.
    
    Returns
    -------
    float
        The predicted price delta (e.g. 0.12 for +12%).
    """
    # Build DataFrame for CatBoost
    if isinstance(input_data, dict):
        df = pd.DataFrame([input_data], columns=FEATURES)
    elif isinstance(input_data, list):
        df = pd.DataFrame([input_data], columns=FEATURES)
    else:
        raise ValueError("Input must be a dict or list matching FEATURES")

    # Make prediction
    preds = _model.predict(df)
    # preds is a list/array of floats
    return float(preds[0])
