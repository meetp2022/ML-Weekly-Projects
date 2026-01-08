from fastapi import FastAPI
import joblib
import pandas as pd

app = FastAPI(title="House Price Prediction API")

# Load model
model = joblib.load("model/model.pkl")

@app.get("/")
def home():
    return {"message": "House Price Prediction API is running!"}

@app.post("/predict")
def predict(data: dict):
    try:
        df = pd.DataFrame([data])
        prediction = model.predict(df)[0]
        return {"predicted_price": float(prediction)}
    except Exception as e:
        import traceback
        print(traceback.format_exc())  # Print detailed error
        return {"error": str(e)}




import requests

WEBHOOK_URL = "http://127.0.0.1:5678/webhook-test/predict"

payload = {
    "city": "Berlin",
    "Location": "Charlottenburg",
    "listed_price": 300000,
    "predicted_price": 375000
}

if __name__ == "__main__":
    try:
        response = requests.post(WEBHOOK_URL, json=payload)
        print("Webhook response:", response.text)
    except Exception as e:
        print("Failed to send data to n8n:", e)

