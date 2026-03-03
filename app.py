from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from fastapi import Request

app = FastAPI()

# Configurar templates
templates = Jinja2Templates(directory="templates")

# Cargar pipeline entrenado
pipeline = joblib.load("loan_pipeline.pkl")


# 👉 NUEVA RUTA RAÍZ
@app.get("/")
def home():
    return {"message": "Loan API is running"}

# 🔹 PASO 4 → RUTA PARA MOSTRAR FORMULARIO
@app.get("/form", response_class=HTMLResponse)
def show_form(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


class LoanRequest(BaseModel):
    Income: float
    Credit_Score: int
    Loan_Amount: float
    DTI_Ratio: float
    Employment_Status: int


@app.post("/predict")
def predict(data: LoanRequest):

    input_df = pd.DataFrame([{
        "Income": data.Income,
        "Credit_Score": data.Credit_Score,
        "Loan_Amount": data.Loan_Amount,
        "DTI_Ratio": data.DTI_Ratio,
        "Employment_Status": data.Employment_Status
    }])

    prediction = pipeline.predict(input_df)[0]

    return {
        "Loan Approval Prediction": int(prediction)
    }