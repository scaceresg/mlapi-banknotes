import pandas as pd
import pickle
from fastapi import FastAPI
from pydantic import BaseModel

# Load model from pickle
with open('banknote_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Class to describe input for the model
class BankNote(BaseModel):
    variance: float
    skewness: float
    curtosis: float
    entropy: float

# Create app object
app = FastAPI()

# Get request (localhost: http://127.0.0.1:8000)
@app.get('/')
def root():
    return {'status': 'OK'}

# Post request
@app.post('/predict')
def authenticate(note:BankNote):
    data = pd.DataFrame([note.dict().values()], columns=note.dict().keys())
    y_pred = model.predict(data)

    if int(y_pred) == 1:
        pred = 'genuine'
    else:
        pred = 'forged'
    return {'prediction': pred, 'class': int(y_pred)}

