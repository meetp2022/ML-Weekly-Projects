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


---

## 🧱 **Project Structure**



Linear_Regression/
│
├── model/
│ └── model.pkl
│
├── Notebooks/
│ └── linear_regression.ipynb
│
├── app.py # FastAPI app
├── requirements.txt # Dependencies
├── Results/ # Example outputs
└── README.md
---

## ⚙️ **How to Run**


```bash
1️⃣ Create and Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

2️⃣ Install Dependencies
pip install -r requirements.txt

3️⃣ Run FastAPI App
uvicorn app:app --reload


The API will start at 👉 http://127.0.0.1:8000

4️⃣ Test /predict Endpoint

Send a POST request:

{
  "city": "Berlin",
  "Location": "Charlottenburg",
  "listed_price": 300000,
  "predicted_price": 375000
}

5️⃣ Connect with n8n

Create a Webhook in n8n

Add HTTP Request node → points to your FastAPI /predict

Add IF node → listed_price < predicted_price

Add Gmail node → send undervaluation alert

✉️ Example Alert Email

Subject: Undervalued Property Detected 🏠

Message:
A property in Berlin - Charlottenburg is undervalued!
Predicted price: €375,000
Listed price: €300,000


📈 Results

✅ Successfully deployed ML model via API
✅ Connected automation workflow using n8n
✅ Received automated Gmail notifications

🧭 Future Enhancements

- Integrate with real real-estate listing APIs (e.g. ImmobilienScout24)

- Add dashboard visualization using Streamlit or Gradio

- Auto-post daily undervalued finds to LinkedIn or Telegram via n8n
