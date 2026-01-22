from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from model.predict import predict
from fastapi.middleware.cors import CORSMiddleware  
import os

app = FastAPI()


origins = ["*"]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
class CDR(BaseModel):
    features: list[float]

@app.post("/api/v1/fraud")
def fraud(cdr: CDR):
    return predict(cdr.dict())

app.mount("/static", StaticFiles(directory="web/public"), name="static")

@app.get("/")
def read_index():
    return FileResponse(os.path.join("web/public", "index.html"))
