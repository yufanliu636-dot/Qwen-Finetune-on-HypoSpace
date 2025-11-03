# Fine_tune_Qwen2.5_QLoRA.py
import os
import json


from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig

# ===============================
# 1️⃣ Basic Settings
# ===============================
os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_path = r"/opt/data/private/Qwen2.5-14B"
dataset_path = r"/opt/data/private/boolean_dataset.json"
output_dir = r"/opt/data/private/qwen2.5-14b-boolean-qlora"

# ===============================
# 2️⃣ Load Tokenizer
# ===============================
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ===============================
# 3️⃣ Load Dataset
# ===============================
print("🔹 Loading dataset...")
with open(dataset_path, "r") as f:
    raw_data = json.load(f)

# Extract data samples
samples = []
for n, data_list in raw_data["datasets_by_n_observations"].items():
    for entry in data_list:
        obs_texts = [o["string"] for o in entry["observations"]]
        user_prompt = (
            "Given the following Boolean observations, generate a Boolean expression "
            "that matches all the outputs.\n\nObservations:\n" + "\n".join(obs_texts)
        )
        gt_exprs = [g["formula"] for g in entry.get("ground_truth_expressions", [])]
        gt_answer = (
            gt_exprs[0] if gt_exprs else "No valid Boolean expression found."
        )

        samples.append({
            "messages": [
                {"role": "system", "content": "You are an expert in Boolean logic."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": gt_answer},
            ]
        })

print(f"✅ Prepared {len(samples)} training samples.")

# Create Hugging Face Dataset
ds = Dataset.from_list(samples)

# ===============================
# 4️⃣ Preprocessing Function
# ===============================
def preprocess(examples, tokenizer, max_len=1024):
    input_ids, labels = [], []

    for s in examples["messages"]:
        sys_msg = s[0]["content"]
        user_msg = s[1]["content"]
        asst_msg = s[2]["content"]

        prompt = f"[System]\n{sys_msg}\n\n[User]\n{user_msg}\n\n[Assistant]\n"
        prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
        target_ids = tokenizer.encode(asst_msg + tokenizer.eos_token, add_special_tokens=False)

        input_seq = prompt_ids + target_ids
        if len(input_seq) > max_len:
            input_seq = input_seq[-max_len:]
            prompt_len = max(0, len(input_seq) - len(target_ids))
        else:
            prompt_len = len(prompt_ids)

        label_seq = [-100] * prompt_len + input_seq[prompt_len:]
        if len(input_seq) < max_len:
            pad_len = max_len - len(input_seq)
            input_seq += [tokenizer.pad_token_id] * pad_len
            label_seq += [-100] * pad_len

        input_ids.append(input_seq)
        labels.append(label_seq)

    return {"input_ids": input_ids, "labels": labels}


print("🔹 Tokenizing dataset...")
tokenized = ds.map(lambda e: preprocess(e, tokenizer), batched=True, remove_columns=["messages"])

# ===============================
# 5️⃣ Load Model (4-bit + QLoRA)
# ===============================
print("🔹 Loading base model in 4-bit...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    
    device_map="auto",
    trust_remote_code=True,
)
model.config.use_cache = False

# Prepare for k-bit training
model = prepare_model_for_kbit_training(model)

print("🔹 Applying QLoRA...")
lora_config = LoraConfig(
    r=64,
    lora_alpha=128,
    target_modules=[
        "q_proj",
        "k_proj",
        "v_proj",
        "o_proj",
        "gate_proj",
        "up_proj",
        "down_proj"
    ],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# ===============================
# 6️⃣ Training Config
# ===============================
print("🔹 Preparing Trainer...")

training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=10,
    max_steps=750,
    learning_rate=2e-4,
    logging_steps=10,
    save_steps=100,
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    group_by_length=True,
    report_to="none",
)

# ===============================
# 7️⃣ Trainer
# ===============================
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    tokenizer=tokenizer,
)

# ===============================
# 8️⃣ Start Fine-Tuning
# ===============================
print("🚀 Start fine-tuning QLoRA model...")
trainer.train()

print("✅ Fine-tuning complete! Model saved at:", output_dir)

