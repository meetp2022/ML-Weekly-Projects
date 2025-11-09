# 🏠 Real Estate Undervaluation Detector (ML + FastAPI + n8n)

This project predicts house prices using a Machine Learning model (Linear Regression) and automates real-time undervaluation detection using **FastAPI** and **n8n**.

When a property listing’s actual price is below the model’s predicted value, an **automated email alert** is sent — just like a smart AI investment assistant. 💡

---

## 🚀 **Project Overview**

Traditional ML projects often stop at a notebook — this one goes further!  
I deployed the trained ML model with **FastAPI** and connected it to **n8n**, a no-code automation tool, to create a *real-world AI workflow*.

### ⚙️ End-to-End Flow
1. **Machine Learning Model**
   - Linear Regression model trained to predict house prices  
   - Saved as `model.pkl` using `joblib`

2. **FastAPI**
   - Serves the model via an API (`/predict` endpoint)
   - Receives property data → returns predicted price

3. **n8n Workflow**
   - Webhook node receives data from FastAPI
   - IF node checks if `listed_price < predicted_price`
   - Sends **email alert** for undervalued properties

4. **Automation Result**
   - A complete ML → API → Automation → Notification loop 🎯

---

## 🧠 **Tech Stack**

| Category | Tools/Frameworks |
|-----------|------------------|
| Machine Learning | Python, Scikit-learn, Pandas, NumPy |
| Model Serving | FastAPI, Uvicorn |
| Automation | n8n (Webhook, HTTP, IF, Gmail nodes) |
| Version Control | Git, GitHub |
| Communication | Gmail API (email alerts) |

---

## 🧩 **Architecture Diagram**
[ Real Estate Data ]
            ↓
[ ML Model (FastAPI API) ]
            ↓
[ n8n Webhook → Condition → Gmail Node ]
            ↓
[ Email Notification for Undervalued Listings ]

