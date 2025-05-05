from huggingface_hub import login

from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForSeq2Seq
from datasets import Dataset
import json

# 加载 instruction 格式的问答数据
with open("data/testqa.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

# 拼接为 LLaMA 风格 prompt 格式
# "### Instruction:\n{instruction}\n\n### Response:\n{output}"
data = [
    {"text": f"### Instruction:\n{item['instruction']}\n\n### Response:\n{item['output']}"} 
    for item in raw_data
]

dataset = Dataset.from_list(data)

# 模型（你可以替换为 meta-llama/Llama-2-7b-hf 或 mistralai/Mistral-7B-Instruct）
model_name = "mistralai/Mistral-7B-Instruct-v0.2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)

# 设置 pad_token
tokenizer.pad_token = tokenizer.eos_token
model.config.pad_token_id = tokenizer.pad_token_id

# Tokenize
def tokenize(example):
    return tokenizer(example["text"], truncation=True, padding="max_length", max_length=512)

dataset = dataset.map(tokenize, batched=True)

data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)

# 训练参数
training_args = TrainingArguments(
    output_dir="trained-model3",
    num_train_epochs=3,
    per_device_train_batch_size=1,
    save_steps=10,
    save_total_limit=1,
    logging_dir="logs",
    logging_steps=5,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    data_collator=data_collator,
)

trainer.train()

trainer.save_model("trained-model3")
tokenizer.save_pretrained("trained-model3")

print("✅ LLaMA / Mistral 指令模型训练完成")
