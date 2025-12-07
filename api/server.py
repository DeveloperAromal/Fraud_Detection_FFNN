from fastapi import FastAPI
from model.predict import predict

from pydantic import BaseModel


app = FastAPI()


class CDR(BaseModel):
    features: list[float]




@app.get("/")
def root():
    return {
                "message": "Fraud detection API"
           }
    
    
@app.post("/api/v1/fraud")
def fraud(cdr: CDR):
    return predict(cdr.dict())