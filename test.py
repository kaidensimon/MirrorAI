import os
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
os.environ["HF_TOKEN"] = "REMOVED"
import json
from datasets import Dataset
from trl import SFTTrainer, SFTConfig
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import torch
from peft import LoraConfig, get_peft_model, TaskType
from format_fine_tuning_data import format_dialogue

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

cache_dir = r"C:\Users\kaide\.cache\huggingface\hub\models--huihui-ai--Huihui-Qwen3.5-2B-abliterated\snapshots"
snapshots = os.listdir(cache_dir)
model_path = os.path.join(cache_dir, snapshots[0])
print(model_path)  #mpute_dtype=torch.bfloat16,

# Find where it ac confirm the path exists

model = AutoModelForCausalLM.from_pretrained(
    model_path,  # use direct path instead of model ID string
    device_map="auto",
    quantization_config=bnb_config,

)

tokenizer = AutoTokenizer.from_pretrained(model_path)
tokenizer.pad_token = tokenizer.eos_token  # required for Qwen




lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
with open("D:\\WeekendSideProjects\\mirrorai\\interview_output.json", "r", encoding="utf-8") as f:
    raw_data = json.load(f)

formatted = format_dialogue(raw_data["dialogue"])
dataset = Dataset.from_list(formatted)

def tokenize(example):
    return tokenizer(
        example["text"],
        truncation=True,
        max_length=512,
        padding="max_length",
    )

tokenized_dataset = dataset.map(tokenize, batched=True)

training_args = SFTConfig(
    output_dir="./mirror-ai-adapter",
    num_train_epochs=5,          # more epochs since dataset is small
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    logging_steps=5,
    save_steps=50,
)

trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

trainer.train()
trainer.save_model("./mirror-ai-adapter")

inputs = tokenizer(
    "<|im_start|>user\nWhat do you think about failure?<|im_end|>\n<|im_start|>assistant\n",
    return_tensors="pt"
).to("cuda")

outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
print(tokenizer.decode(outputs[0], skip_special_tokens=True))