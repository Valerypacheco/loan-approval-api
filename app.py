from fastapi import FastAPI, Request
from pydantic import BaseModel
import joblib
import pandas as pd
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates



app = FastAPI()

# Configure templates
templates = Jinja2Templates(directory="templates")

# Load trained pipeline
pipeline = joblib.load("model/loan_pipeline.pkl")


# # Root endpoint
# @app.get("/")
# def home():
#     return {
#         "message": "Loan Approval API is running",
#         "status": "success"
#     }

# Route to display form
# @app.get("/form", response_class=HTMLResponse)
# def show_form(request: Request):
#     return templates.TemplateResponse("index.html", {"request": request})
@app.get("/", response_class=HTMLResponse)
def home(request: Request):
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
# http://127.0.0.1:8000/form