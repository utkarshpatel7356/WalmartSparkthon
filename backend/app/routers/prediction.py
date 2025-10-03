from fastapi import APIRouter, Request, Form
from fastapi.responses import JSONResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd
import torch
import torch.nn as nn
import pickle

from ml_models.catboost_price_delta_model import predict_price_delta
from ml_models.acquisitions import predict_acquisition_quantity

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")


### ------------------------ PRICE DELTA SECTION ------------------------ ###
class PriceDeltaRequest(BaseModel):
    location: str
    effective_demand: float
    stock_level: float
    competitor_price: float
    weather: str
    product_name: str
    category: str
    base_price: float
    shelf_life_days: float

# @router.post("/predict/price-delta")
@router.post("/price-delta")
def handle_price_delta(request_data: PriceDeltaRequest):
    try:
        input_data = request_data.dict()
        pred = predict_price_delta(input_data) * 100
        return {"prediction": round(pred, 2)}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

### ------------------------ ACQUISITION SECTION ------------------------ ###
class AcquisitionRequest(BaseModel):
    demand: float
    stock_level: float
    shelf_life_days: float
    price: float
    spoilage_cost: float

# @router.post("/predict/acquisition")
@router.post("/acquisition")
def handle_acquisition(request_data: AcquisitionRequest):
    try:
        input_data = request_data.dict()
        quantity = predict_acquisition_quantity(input_data)
        return {"prediction": quantity}
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

### ------------------------ DEMAND FORECAST SECTION ------------------------ ###
# Load TabTransformer model and artifacts on startup
class TabTransformer(nn.Module):
    def __init__(self, num_numerical_features, categorical_embeddings, dim_embedding=64, num_heads=8, num_layers=4, dropout=0.2):
        super(TabTransformer, self).__init__()
        self.cat_embeddings = nn.ModuleDict({
            cat: nn.Embedding(size, dim_embedding)
            for cat, size in categorical_embeddings.items()
        })
        self.numerical_linear = nn.Linear(num_numerical_features, dim_embedding)
        total_features = len(categorical_embeddings) + 1
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding, nhead=num_heads, dim_feedforward=dim_embedding * 4, dropout=dropout, batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.output_layers = nn.Sequential(
            nn.Linear(dim_embedding * total_features, dim_embedding * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_embedding * 2, dim_embedding),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(dim_embedding, 1)
        )

    def forward(self, numerical_features, categorical_features):
        batch_size = numerical_features.size(0)
        embeddings = [self.numerical_linear(numerical_features).unsqueeze(1)]
        for cat in categorical_features:
            emb = self.cat_embeddings[cat](categorical_features[cat])
            embeddings.append(emb.unsqueeze(1))
        x = torch.cat(embeddings, dim=1)
        x = self.transformer(x)
        x = x.view(batch_size, -1)
        return self.output_layers(x).squeeze(-1)

model = None
artifacts = None

def load_tabtransformer():
    global model, artifacts
    with open("ml_models/model_artifacts.pkl", "rb") as f:
        artifacts = pickle.load(f)
    model = TabTransformer(**artifacts["model_config"])
    model.load_state_dict(torch.load("ml_models/tabtransformer_model.pth", map_location="cpu"))
    model.eval()

def get_valid_values():
    label_encoders = artifacts["label_encoders"]
    return {col: list(enc.classes_) for col, enc in label_encoders.items()}

def predict_demand_transformer(data):
    df = pd.DataFrame([data])
    le = artifacts["label_encoders"]
    scaler = artifacts["scaler"]
    cat_cols = artifacts["categorical_cols"]
    num_cols = artifacts["numerical_cols"]

    for col in cat_cols:
        df[col] = le[col].transform(df[col])
    df[num_cols] = scaler.transform(df[num_cols])

    num_tensor = torch.FloatTensor(df[num_cols].values)
    cat_tensor = {col: torch.LongTensor(df[col].values) for col in cat_cols}

    with torch.no_grad():
        pred = model(num_tensor, cat_tensor)
        return float(pred.numpy()[0])

@router.post("/predict/demand")
async def handle_demand_predict(
    request: Request,
    location: str = Form(...),
    stock_level: str = Form(...),
    competitor_price: str = Form(...),
    weather: str = Form(...),
    product_name: str = Form(...),
    base_price: str = Form(...),
    shelf_life_days: str = Form(...),
    category: str = Form(...)
):
    try:
        data = {
            "location": location,
            "stock_level": int(stock_level),
            "competitor_price": float(competitor_price),
            "weather": weather,
            "product_name": product_name,
            "base_price": float(base_price),
            "shelf_life_days": int(shelf_life_days),
            "category": category,
        }
        prediction = predict_demand_transformer(data)
        return JSONResponse(content={"prediction": round(prediction, 2)})
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

### ------------------------ MAIN PREDICTION PAGE ------------------------ ###
@router.get("/prediction", name="Prediction Dashboard")
def render_prediction_page(request: Request):
    values = get_valid_values()
    return templates.TemplateResponse("prediction.html", {
        "request": request,
        "valid_values": values
    })

# Load model at import time
load_tabtransformer()
