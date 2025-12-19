from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model.predict import predict
import os

app = FastAPI()

class CDR(BaseModel):
    features: list[float]

@app.post("/api/v1/fraud")
def fraud(cdr: CDR):
    return predict(cdr.dict())

app.mount("/static", StaticFiles(directory="web/public"), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join("web/public", "index.html"))
