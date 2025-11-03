# Fine_tune_Qwen2.5_QLoRA_final.py
import os
import json
import random
import torch
import gc
from transformers import TrainerCallback

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
model_path = r"/opt/data/private/Models/Qwen2.5-14B"
dataset_path = r"/opt/data/private/3d/grid_observation_dataset_2000.json"
output_dir = r"/opt/data/private/QLoRA"

# 清理内存
torch.cuda.empty_cache()
gc.collect()

# ===============================
# 2️⃣ Load Tokenizer
# ===============================
print("🔹 Loading tokenizer...")
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ===============================
# 3️⃣ Load Dataset
# ===============================
print("🔹 Loading dataset...")
with open(dataset_path, "r") as f:
    raw_data = json.load(f)

samples = []
for observation_set in raw_data["observation_sets"][:800]:
    observation_string = observation_set["observation"]
    ground_truth_structures = observation_set["ground_truth_structures"]
    
    # 网格表示
    grid_representation = ""
    obs_chars = list(observation_string)
    for i in range(0, 9, 3):
        grid_representation += " ".join(obs_chars[i:i+3]) + "\n"
    grid_representation = grid_representation.strip()
    
    if ground_truth_structures:
        selected_structure = random.choice(ground_truth_structures)
        layers = selected_structure["layers"]
        
        structure_representation = "Possible 3D structure:\n"
        for idx, layer in enumerate(layers, 1):
            layer_grid = ""
            layer_chars = list(layer)
            for i in range(0, 9, 3):
                layer_grid += " ".join(layer_chars[i:i+3]) + "\n"
            structure_representation += f"Layer {idx}:\n{layer_grid.strip()}\n"
        
        user_prompt = f"Observation:\n{grid_representation}\nWhat 3D structure could produce this?"

        samples.append({
            "messages": [
                {"role": "system", "content": "Expert in spatial reasoning."},
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": structure_representation.strip()},
            ]
        })

print(f"✅ Prepared {len(samples)} samples.")
ds = Dataset.from_list(samples)

# ===============================
# 4️⃣ Preprocessing
# ===============================
def preprocess(examples, tokenizer, max_len=512):  # 进一步减少序列长度
    texts = []
    for s in examples["messages"]:
        sys_msg = s[0]["content"]
        user_msg = s[1]["content"]
        asst_msg = s[2]["content"]
        
        # 使用更简洁的格式
        text = f"<|im_start|>system\n{sys_msg}<|im_end|>\n<|im_start|>user\n{user_msg}<|im_end|>\n<|im_start|>assistant\n{asst_msg}<|im_end|>"
        texts.append(text)
    
    # 使用tokenizer的批量处理
    tokenized = tokenizer(
        texts,
        truncation=True,
        padding=True,
        max_length=max_len,
        return_tensors=None,
    )
    
    # 对于因果LM，labels与input_ids相同
    tokenized["labels"] = tokenized["input_ids"].copy()
    
    return tokenized

print("🔹 Tokenizing...")
tokenized = ds.map(
    lambda e: preprocess(e, tokenizer), 
    batched=True, 
    remove_columns=ds.column_names
)

# ===============================
# 5️⃣ Load Model
# ===============================
print("🔹 Loading model...")

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_path,
    quantization_config=bnb_config,
    device_map="auto",
    trust_remote_code=True,
    dtype=torch.bfloat16,  # 使用dtype而不是torch_dtype
)

model.config.use_cache = False
model = prepare_model_for_kbit_training(model)

# LoRA配置
lora_config = LoraConfig(
    r=64,  # 使用更小的秩
    lora_alpha=128,
    target_modules=[
        "q_proj",
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
# 6️⃣ 修复的回调类
# ===============================
class MemoryMonitorCallback(TrainerCallback):
    def on_step_end(self, args, state, control, **kwargs):
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    def on_log(self, args, state, control, logs=None, **kwargs):
        if torch.cuda.is_available() and state.global_step % 25 == 0:
            allocated = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            print(f"🎯 Step {state.global_step}: GPU Memory - Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")

# ===============================
# 7️⃣ Training Arguments
# ===============================
training_args = TrainingArguments(
    output_dir=output_dir,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    warmup_steps=50,
    max_steps=700,  
    learning_rate=1e-4,
    logging_steps=25,
    save_steps=100,
    bf16=True,
    optim="paged_adamw_8bit",
    gradient_checkpointing=True,
    dataloader_pin_memory=False,
    remove_unused_columns=False,
    report_to="none",
    dataloader_num_workers=0,
    eval_strategy="no",
    gradient_checkpointing_kwargs={"use_reentrant": False},
)

# ===============================
# 8️⃣ 创建Trainer
# ===============================
print("🚀 Creating trainer...")
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized,
    processing_class=tokenizer,  # 使用processing_class而不是tokenizer
    callbacks=[MemoryMonitorCallback()],
)

print("📊 Training Configuration:")
print(f"   - Batch size: {training_args.per_device_train_batch_size}")
print(f"   - Gradient accumulation: {training_args.gradient_accumulation_steps}")
print(f"   - Effective batch size: {training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps}")
print(f"   - Max length: 4096")
print(f"   - LoRA rank: 64")

# ===============================
# 9️⃣ 开始训练
# ===============================
print("🔥 Starting training...")
try:
    trainer.train()
    trainer.save_model()
    print("✅ Training completed successfully!")
    
except RuntimeError as e:
    if "out of memory" in str(e):
        print("❌ OOM detected! Trying more aggressive optimizations...")
        
        # 保存当前状态
        torch.cuda.empty_cache()
        gc.collect()
        
        # 重新配置
        training_args.per_device_train_batch_size = 1
        training_args.gradient_accumulation_steps = 2
        training_args.max_steps = 300
        
        print("🔄 Restarting with minimal configuration...")
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized,
            processing_class=tokenizer,
            callbacks=[MemoryMonitorCallback()],
        )
        trainer.train()
        trainer.save_model()
        print("✅ Training completed with minimal configuration!")
    else:
        print(f"❌ Unexpected error: {e}")
        raise e

# ===============================
# 🔟 测试函数
# ===============================
def test_model():
    print("\n🔹 Testing fine-tuned model...")
    try:
        # 切换到评估模式
        model.eval()
        
        test_observation = "100010001"
        test_grid = "\n".join([" ".join(test_observation[i:i+3]) for i in range(0, 9, 3)])

        test_prompt = f"<|im_start|>system\nExpert in spatial reasoning.<|im_end|>\n<|im_start|>user\nObservation:\n{test_grid}\nWhat 3D structure could produce this?<|im_end|>\n<|im_start|>assistant\n"

        inputs = tokenizer(test_prompt, return_tensors="pt", max_length=256, truncation=True)
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=150,
                do_sample=True,
                temperature=0.7,
                pad_token_id=tokenizer.eos_token_id
            )

        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print("🤖 Model Response:")
        print(response)
        
    except Exception as e:
        print(f"❌ Test failed: {e}")

# 运行测试
test_model()

print(f"🎉 All done! Model saved to: {output_dir}")