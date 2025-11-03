

import os, json, random, torch, gc
from datasets import Dataset
from transformers import (
    AutoTokenizer, AutoModelForCausalLM,
    TrainingArguments, Trainer, BitsAndBytesConfig,
    DataCollatorForSeq2Seq, TrainerCallback
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# ========== 可改配置（无需命令行） ==========
CONFIG = {
    # 本地模型与数据路径
    "model_path": r"Your model path",
    "dataset_path": r"Your dataset path",
    "output_dir": r"Your output path",
    "max_length": 4096,
    # LoRA（你的要求）
    "lora_r": 64,
    "lora_alpha": 128,
    "lora_dropout": 0.05,

    # 训练（你的要求）
    "per_device_train_batch_size": 1,
    "gradient_accumulation_steps": 4,
    "learning_rate": 1e-4,     # 学习率 1e-4
    "warmup_steps": 50,
    "max_steps": 700,          # 固定步数训练
    "save_steps": 100,
    "logging_steps": 10,
    "eval_strategy": "steps", 
    "eval_steps": 100,
    "eval_split_ratio": 0.1,   # 验证集比例
}
# =========================================

os.environ["TOKENIZERS_PARALLELISM"] = "false"
torch.cuda.empty_cache(); gc.collect()

# 1) Tokenizer
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(CONFIG["model_path"], trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# 2) 读取新数据集（instruction/input/output），转为 Qwen chat 文本
print("🔹 Loading dataset...")
with open(CONFIG["dataset_path"], "r", encoding="utf-8") as f:
    raw_list = json.load(f)

def to_chat_text(instruction: str, user_input: str, output: str) -> str:
    # 系统消息改为英文
    sys_msg = "You are a strict causal inference assistant. Please list all possible causal graphs based on the observations, maintaining the same output format as the examples (multiple lines enumeration)."
    user_msg = (instruction.strip() + "\n\n" + user_input.strip()).strip()
    asst_msg = output.strip()
    return (
        f"<|im_start|>system\n{sys_msg}<|im_end|>\n"
        f"<|im_start|>user\n{user_msg}<|im_end|>\n"
        f"<|im_start|>assistant\n{asst_msg}<|im_end|>"
    )

samples = []
for ex in raw_list:
    ins = ex.get("instruction", "")
    inp = ex.get("input", "")
    out = ex.get("output", "")
    if not (ins and inp and out):
        continue
    samples.append({"text": to_chat_text(ins, inp, out)})

print(f"✅ Prepared {len(samples)} samples.")

# 先分割数据集，再分别处理
raw_ds = Dataset.from_list(samples)
train_test_split = raw_ds.train_test_split(
    test_size=CONFIG["eval_split_ratio"], 
    seed=42
)

def preprocess_function(batch):
    toks = tokenizer(
        batch["text"],
        truncation=True,
        padding=False,
        max_length=CONFIG["max_length"],
        return_tensors=None
    )
    toks["labels"] = toks["input_ids"].copy()
    return toks

print("🔹 Tokenizing...")
train_dataset = train_test_split["train"].map(
    preprocess_function, batched=True, remove_columns=raw_ds.column_names, batch_size=1000
)
eval_dataset = train_test_split["test"].map(
    preprocess_function, batched=True, remove_columns=raw_ds.column_names, batch_size=1000
)

print(f"✅ Train: {len(train_dataset)}, Eval: {len(eval_dataset)}")

data_collator = DataCollatorForSeq2Seq(
    tokenizer=tokenizer,
    padding=True,
    max_length=CONFIG["max_length"],
    return_tensors="pt"
)

# 3) 模型 + QLoRA
print("🔹 Loading model with 4-bit QLoRA...")
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    CONFIG["model_path"],
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,
)
model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

lora_cfg = LoraConfig(
    r=CONFIG["lora_r"],
    lora_alpha=CONFIG["lora_alpha"],
    lora_dropout=CONFIG["lora_dropout"],
    target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)
model.print_trainable_parameters()

# 4) 改进的内存监控
class ImprovedMemoryCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and state.global_step % 50 == 0:
            torch.cuda.empty_cache()

# 5) 完整的训练参数
training_args = TrainingArguments(
    output_dir=CONFIG["output_dir"],
    per_device_train_batch_size=CONFIG["per_device_train_batch_size"],
    gradient_accumulation_steps=CONFIG["gradient_accumulation_steps"],
    learning_rate=CONFIG["learning_rate"],
    warmup_steps=CONFIG["warmup_steps"],
    max_steps=CONFIG["max_steps"],
    logging_steps=CONFIG["logging_steps"],
    save_steps=CONFIG["save_steps"],
    eval_steps=CONFIG["eval_steps"],
    eval_strategy=CONFIG["eval_strategy"],
    save_strategy="steps",  # 与eval_strategy保持一致
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    bf16=True,
    optim="adamw_torch",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    dataloader_num_workers=0,
    dataloader_drop_last=True,
    report_to="none",
)

# 6) 完整的Trainer
print("🚀 Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,  # 添加验证集
    tokenizer=tokenizer,
    data_collator=data_collator,
    callbacks=[ImprovedMemoryCallback()],
)

print("📊 Training Configuration:")
print(f"   - BSZ: {training_args.per_device_train_batch_size} x GA {training_args.gradient_accumulation_steps}")
print(f"   - LR: {training_args.learning_rate}")
print(f"   - Steps: max={CONFIG['max_steps']}, save_steps={CONFIG['save_steps']}, eval_steps={CONFIG['eval_steps']}")
print(f"   - LoRA r/alpha: {CONFIG['lora_r']}/{CONFIG['lora_alpha']}")
print(f"   - Max length: {CONFIG['max_length']}")
print(f"   - Dataset: train={len(train_dataset)}, eval={len(eval_dataset)}")

print("🔥 Start training...")
trainer.train()
trainer.save_model()
tokenizer.save_pretrained(CONFIG["output_dir"])

print(f"🎉 Done. Model saved to: {CONFIG['output_dir']}")