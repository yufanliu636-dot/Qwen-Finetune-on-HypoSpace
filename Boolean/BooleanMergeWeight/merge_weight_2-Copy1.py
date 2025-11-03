# merge_weight_final.py
import os
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

print("🔹 开始合并 LoRA 权重到基础模型中...")

# 设置路径 - 使用最终的检查点
base_model_path = r"/opt/data/private/Qwen2.5-14B"
adapter_path = r"/opt/data/private/causal/checkpoint-200"  # 使用最终检查点
merged_model_path = r"/opt/data/private/causal/checkpoint-200"

try:
    # 检查检查点目录
    print(f"🔹 使用检查点: {adapter_path}")
    if not os.path.exists(adapter_path):
        print(f"❌ 检查点路径不存在: {adapter_path}")
        exit(1)
    
    config_file = os.path.join(adapter_path, "adapter_config.json")
    if not os.path.exists(config_file):
        print(f"❌ 在检查点中找不到 adapter_config.json")
        exit(1)
    
    print("🔹 加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, trust_remote_code=True)
    
    print("🔹 加载基础模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )
    
    print("🔹 从检查点加载适配器并合并权重...")
    model = PeftModel.from_pretrained(base_model, adapter_path)
    merged_model = model.merge_and_unload()
    
    print(f"🔹 保存合并后的模型到: {merged_model_path}")
    merged_model.save_pretrained(merged_model_path)
    tokenizer.save_pretrained(merged_model_path)
    
    print("✅ 权重合并完成！")
    print(f"   基础模型: {base_model_path}")
    print(f"   适配器: {adapter_path}") 
    print(f"   合并后模型: {merged_model_path}")
    
except Exception as e:
    print(f"❌ 权重合并失败: {e}")
    import traceback
    traceback.print_exc()