from fastapi import APIRouter, Request, Form, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import pickle
import os
from typing import Dict, Any

router = APIRouter()
templates = Jinja2Templates(directory="frontend/templates")

# TabTransformer Model Class
class TabTransformer(nn.Module):
    """TabTransformer model for tabular data prediction"""
    
    def __init__(self, num_numerical_features, categorical_embeddings, dim_embedding=64, num_heads=8, num_layers=4, dropout=0.2):
        super(TabTransformer, self).__init__()
        
        # Store parameters
        self.num_numerical_features = num_numerical_features
        self.categorical_embeddings = categorical_embeddings
        self.dim_embedding = dim_embedding
        
        # Embedding layers for categorical features
        self.cat_embeddings = nn.ModuleDict()
        for cat_name, vocab_size in categorical_embeddings.items():
            self.cat_embeddings[cat_name] = nn.Embedding(vocab_size, dim_embedding)
        
        # Linear layer for numerical features
        self.numerical_linear = nn.Linear(num_numerical_features, dim_embedding)
        
        # Total input dimension for transformer
        total_features = len(categorical_embeddings) + 1  # +1 for numerical features
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim_embedding, 
            nhead=num_heads, 
            dim_feedforward=dim_embedding * 4,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        # Output layers for regression
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
        embeddings = []
        
        # Process numerical features
        num_emb = self.numerical_linear(numerical_features)
        embeddings.append(num_emb.unsqueeze(1))  # Add sequence dimension
        
        # Process categorical features
        for cat_name, cat_values in categorical_features.items():
            cat_emb = self.cat_embeddings[cat_name](cat_values)
            embeddings.append(cat_emb.unsqueeze(1))  # Add sequence dimension
        
        # Concatenate all embeddings
        x = torch.cat(embeddings, dim=1)  # Shape: [batch_size, num_features, dim_embedding]
        
        # Apply transformer
        x = self.transformer(x)  # Shape: [batch_size, num_features, dim_embedding]
        
        # Flatten for final layers
        x = x.view(batch_size, -1)  # Shape: [batch_size, num_features * dim_embedding]
        
        # Final prediction
        output = self.output_layers(x)
        return output.squeeze(-1)  # Remove last dimension for regression

# Global model instance
model_instance = None
artifacts = None

def load_model():
    """Load the trained model and preprocessing artifacts"""
    global model_instance, artifacts
    
    try:
        # Load artifacts
        artifacts_path = 'ml_models/model_artifacts.pkl'
        with open(artifacts_path, 'rb') as f:
            artifacts = pickle.load(f)
        
        # Initialize model
        model_config = artifacts['model_config']
        model_instance = TabTransformer(**model_config)
        
        # Load model weights
        model_path = 'ml_models/tabtransformer_model.pth'
        model_instance.load_state_dict(torch.load(model_path, map_location='cpu'))
        model_instance.eval()
        
        print("✅ Model loaded successfully!")
        
    except Exception as e:
        print(f"❌ Error loading model: {str(e)}")
        raise e

# Load model on startup
try:
    load_model()
except Exception as e:
    print(f"Warning: Could not load model: {e}")

def get_valid_values():
    """Get valid values for categorical features"""
    if artifacts is None:
        return {}
    
    label_encoders = artifacts['label_encoders']
    valid_values = {}
    
    for col, encoder in label_encoders.items():
        valid_values[col] = list(encoder.classes_)
    
    return valid_values

def predict_single(location, stock_level, competitor_price, weather, 
                  product_name, base_price, shelf_life_days, category):
    """
    Predict effective demand for a single input
    """
    if model_instance is None or artifacts is None:
        raise ValueError("Model not loaded")
    
    # Create input dataframe
    input_data = pd.DataFrame({
        'location': [location],
        'stock_level': [int(stock_level)],
        'competitor_price': [float(competitor_price)],
        'weather': [weather],
        'product_name': [product_name],
        'base_price': [float(base_price)],
        'shelf_life_days': [int(shelf_life_days)],
        'category': [category]
    })
    
    # Get preprocessing artifacts
    label_encoders = artifacts['label_encoders']
    scaler = artifacts['scaler']
    categorical_cols = artifacts['categorical_cols']
    numerical_cols = artifacts['numerical_cols']
    
    # Preprocess input data
    processed_data = input_data.copy()
    
    # Encode categorical features
    for col in categorical_cols:
        if col in processed_data.columns:
            processed_data[col] = label_encoders[col].transform(processed_data[col])
    
    # Scale numerical features
    processed_data[numerical_cols] = scaler.transform(processed_data[numerical_cols])
    
    # Convert to tensors
    num_features = torch.FloatTensor(processed_data[numerical_cols].values)
    cat_features = {}
    for col in categorical_cols:
        cat_features[col] = torch.LongTensor(processed_data[col].values)
    
    # Make predictions
    with torch.no_grad():
        prediction = model_instance(num_features, cat_features)
        return float(prediction.numpy()[0])

@router.get("/trnf", response_class=HTMLResponse)
async def trnf_page(request: Request):
    """Render the transformer prediction page"""
    valid_values = get_valid_values()
    
    return templates.TemplateResponse("trnf.html", {
        "request": request,
        "valid_values": valid_values
    })

@router.post("/trnf/predict")
async def predict_demand(
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
    """Handle prediction request"""
    try:
        # Validate inputs
        stock_level = int(stock_level)
        competitor_price = float(competitor_price)
        base_price = float(base_price)
        shelf_life_days = int(shelf_life_days)
        
        # Make prediction
        prediction = predict_single(
            location=location,
            stock_level=stock_level,
            competitor_price=competitor_price,
            weather=weather,
            product_name=product_name,
            base_price=base_price,
            shelf_life_days=shelf_life_days,
            category=category
        )
        
        # Prepare input data for display
        input_data = {
            'location': location,
            'stock_level': stock_level,
            'competitor_price': competitor_price,
            'weather': weather,
            'product_name': product_name,
            'base_price': base_price,
            'shelf_life_days': shelf_life_days,
            'category': category
        }
        
        valid_values = get_valid_values()
        
        return templates.TemplateResponse("trnf.html", {
            "request": request,
            "valid_values": valid_values,
            "prediction": round(prediction, 2),
            "input_data": input_data,
            "success": True
        })
        
    except ValueError as e:
        valid_values = get_valid_values()
        return templates.TemplateResponse("trnf.html", {
            "request": request,
            "valid_values": valid_values,
            "error": f"Invalid input: {str(e)}",
            "success": False
        })
    except Exception as e:
        valid_values = get_valid_values()
        return templates.TemplateResponse("trnf.html", {
            "request": request,
            "valid_values": valid_values,
            "error": f"Prediction error: {str(e)}",
            "success": False
        })