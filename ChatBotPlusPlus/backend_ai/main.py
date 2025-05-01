from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class AskRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: AskRequest):
    question = request.question
    # 暂时用固定回答
    return {"answer": f"你问的是：{question}，但我还没接上AI模型 😅"}