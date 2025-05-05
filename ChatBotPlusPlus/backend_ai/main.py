from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

app = FastAPI()

# 允许 Qt 请求 FastAPI（前后端跨域允许）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

tokenizer = AutoTokenizer.from_pretrained("../trained-model2")
model = AutoModelForCausalLM.from_pretrained("../trained-model2")
model.eval()

# class AskRequest(BaseModel):
#     question: str

@app.post("/ask")
async def ask(request: Request):
    data = await request.json()
    question = data.get("question", "")

    input_text = question + tokenizer.eos_token
    input_ids = tokenizer.encode(input_text, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_length=128,
            pad_token_id=tokenizer.eos_token_id,
            do_sample=False,
            num_beams=5
            # top_k=50,
            # top_p=0.95
        )

    answer = tokenizer.decode(output_ids[:, input_ids.shape[-1]:][0], skip_special_tokens=True)

    return {"answer": answer}
    