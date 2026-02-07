from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import pandas as pd
import json

#---Configuration---
MODEL_PATH = 'models/churn_model_v1.pkl'
META_PATH = 'models/churn_model_v1_meta.json'

# Define input schema(Data validation  using Pydantic)
class CustomerData(BaseModel):
    CreditScore: int
    Age: int
    Balance: float
    Tenure: int

#Initialize the FastAPI app
app = FastAPI(title="Customer Churn Prediction sAPI", version="1.0")

#global variables
model = None
metadata = None 

#Load model on startup(only once, not on every request)
@app.on_event("startup")
def load_artifacts():
    global model, metadata
    #check if model exists
    if os.path.exists(MODEL_PATH) and os.path.exists(META_PATH):
        print("Loading model and metadata...")
        model = joblib.load(MODEL_PATH)
        with open(META_PATH, 'r') as f:
            metadata = json.load(f)
        print("Model and metadata loaded successfully.")
    else:
        print("Model or metadata not found. Please train the model first.")

#Creating Endpoints

@app.get("/health")
def health_check():
    return {"status": "ok", "model_loaded": model is not None}

@app.get("/model-info")
def get_model_info():
    if metadata:
        return metadata
    else:
        raise HTTPException(status_code=404, detail="Model metadata not found")

@app.post("/predict")
def predict(data: CustomerData):
    if not model:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    #Convert input data to DataFrame
    input_df = pd.DataFrame([data.dict()])
    #Make prediction
    try:
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0].tolist()
        
        return {
            "churn_prediction": int(prediction),
            "churn_probability": probability[1],
            "model_version": metadata.get("version") if metadata else "unknown" 
        } 
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"prediction Error: {str(e)}")
    
#Run the app using: uvicorn main:app --reload
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

