# uvicorn main:app --reload               # Run the server

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from router import explanation_settai

app = FastAPI()

origins = [
    "http://localhost",
    "http://localhost:3000",
    "http://localhost:8080",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

@app.get("/")
def read_root(query: str):
    l = len(query)
    return {
        "Score": 50,
        "Length": l
        } 

app.include_router(explanation_settai.router, prefix="/chat")