import traceback
import requests
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import re


from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

class LLMInterface(ABC):
    """Abstract base class for all LLM interfaces."""

    @abstractmethod
    def query(self, prompt: str) -> str:
        """Send a query to the LLM and return the response text."""
        pass

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """Optional extension: return response + token usage info."""
        return {
            "response": self.query(prompt),
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
            "cost": 0.0,
        }

    @abstractmethod
    def get_name(self) -> str:
        """Return a human-readable name for the model."""
        pass

    def get_model_pricing(self) -> Dict[str, float]:
        """Default pricing per 1M tokens (USD)."""
        return {"input": 0.0, "output": 0.0}


# =======================================================================
# 🔹 OpenRouter API
# =======================================================================

class OpenRouterLLM(LLMInterface):
    """LLM interface using OpenRouter unified API gateway."""

    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 40960,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        if not api_key:
            raise ValueError("OpenRouter API key is required")

        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }

    def query(self, prompt: str) -> str:
        result = self.query_with_usage(prompt)
        return result["response"]

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert in Boolean logic and causal inference."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            resp = requests.post(url, headers=self.headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            usage = data.get("usage", {})
            usage_data = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

            pricing = self.get_model_pricing()
            cost = (usage_data["prompt_tokens"] * pricing["input"] +
                    usage_data["completion_tokens"] * pricing["output"]) / 1_000_000

            return {
                "response": data["choices"][0]["message"]["content"],
                "usage": usage_data,
                "cost": cost,
            }

        except Exception as e:
            return {
                "response": f"Error querying OpenRouter: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": 0.0,
            }

    def get_name(self) -> str:
        return f"OpenRouter({self.model})"

    def get_model_pricing(self) -> Dict[str, float]:
        return {
            "input": 3.0,
            "output": 15.0
        }


# =======================================================================
# 🔹 OpenAI API
# =======================================================================

class OpenAILLM(LLMInterface):
    """LLM interface for OpenAI models (e.g., GPT-4, GPT-4o, GPT-5)."""

    def __init__(
        self,
        model: str = "gpt-4o",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        try:
            import openai
        except ImportError:
            raise ImportError("Please install the openai package: pip install openai")

        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key is required")

        self.client = openai.OpenAI(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query(self, prompt: str) -> str:
        return self.query_with_usage(prompt)["response"]

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        try:
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are an expert in Boolean logic and reasoning."},
                    {"role": "user", "content": prompt},
                ],
                max_output_tokens=self.max_tokens,
            )

            # Extract text and usage
            text = getattr(resp, "output_text", str(resp))
            usage = getattr(resp, "usage", None)
            in_tok = getattr(usage, "input_tokens", 0)
            out_tok = getattr(usage, "output_tokens", 0)
            tot_tok = getattr(usage, "total_tokens", in_tok + out_tok)

            pricing = self.get_model_pricing()
            cost = (in_tok * pricing["input"] + out_tok * pricing["output"]) / 1_000_000

            return {
                "response": text,
                "usage": {
                    "prompt_tokens": in_tok,
                    "completion_tokens": out_tok,
                    "total_tokens": tot_tok,
                },
                "cost": cost,
            }

        except Exception as e:
            traceback.print_exc()
            return {
                "response": f"Error querying OpenAI: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": 0.0,
            }

    def get_name(self) -> str:
        return f"OpenAI({self.model})"

    def get_model_pricing(self) -> Dict[str, float]:
        return {"input": 2.5, "output": 10.0}


# =======================================================================
# 🔹 DeepSeek API
# =======================================================================



class LocalQwenLLM(LLMInterface):
    """Local Qwen model with API-style interface like OpenAI/Anthropic."""

    def __init__(
        self,
        model_path: str = r"C:\Users\2049~\Desktop\deepseek_finetune\deepseek-coder-1.3b-base",
        device: str = "cuda",
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens

        # 加载 tokenizer 和模型
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map="auto" if device == "cuda" else None,
            torch_dtype=torch.float16 if device == "cuda" else torch.float32
        )

    def query(self, prompt: str) -> str:
        """返回纯文本响应"""
        return self.query_with_usage(prompt)["response"]

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """返回响应 + token 使用情况 + cost"""
        try:
            # 模拟 messages 风格
            messages = [{"role": "user", "content": prompt}]
            prompt_text = "\n".join([m["content"] for m in messages])

            # 编码输入
            inputs = self.tokenizer(prompt_text, return_tensors="pt").to(self.device)

            # 生成输出
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True,
                    temperature=self.temperature
                )

            # 解码文本
            text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            # 手动计算 token 使用
            total_tokens = outputs.shape[1]
            prompt_tokens = inputs["input_ids"].shape[1]
            completion_tokens = total_tokens - prompt_tokens

            usage_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }

            return {
                "response": text,
                "usage": usage_data,
                "cost": 0.0  # 本地模型不收费
            }

        except Exception as e:
            return {
                "response": f"Error querying Local Qwen: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": 0.0
            }

    def get_name(self) -> str:
        return f"LocalQwen({self.model_path})"

    def get_model_pricing(self) -> Dict[str, float]:
        return {"input": 0.0, "output": 0.0}





class DeepSeekLLM(LLMInterface):
    """Direct API integration for DeepSeek models."""

    def __init__(
        self,
        model: str = "deepseek-chat",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096,
        base_url: str = "https://api.deepseek.com/v1"
    ):
        if not api_key:
            raise ValueError("DeepSeek API key is required")

        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }

    def query(self, prompt: str) -> str:
        result = self.query_with_usage(prompt)
        return result["response"]

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an expert in Boolean logic. "
                            "Given truth table observations, generate logically valid "
                            "Boolean expressions that exactly match all observations."
                        )
                    },
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
            }

            resp = requests.post(url, headers=self.headers, json=payload)
            resp.raise_for_status()
            data = resp.json()

            usage = data.get("usage", {})
            usage_data = {
                "prompt_tokens": usage.get("prompt_tokens", 0),
                "completion_tokens": usage.get("completion_tokens", 0),
                "total_tokens": usage.get("total_tokens", 0),
            }

            # DeepSeek approximate pricing (USD per 1M tokens)
            cost = (usage_data["prompt_tokens"] * 0.4 +
                    usage_data["completion_tokens"] * 2.0) / 1_000_000

            return {
                "response": data["choices"][0]["message"]["content"],
                "usage": usage_data,
                "cost": cost,
            }

        except Exception as e:
            return {
                "response": f"Error querying DeepSeek: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": 0.0,
            }

    def get_name(self) -> str:
        return f"DeepSeek({self.model})"

    def get_model_pricing(self) -> Dict[str, float]:
        return {"input": 0.4, "output": 2.0}


# =======================================================================
# 🔹 Anthropic (Claude) API
# =======================================================================

class AnthropicLLM(LLMInterface):
    """Interface for Anthropic Claude models."""

    def __init__(
        self,
        model: str = "claude-3.5-sonnet-20241022",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 4096
    ):
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic: pip install anthropic")

        if not api_key:
            import os
            api_key = os.getenv("ANTHROPIC_API_KEY")
            if not api_key:
                raise ValueError("Anthropic API key is required")

        self.client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens

    def query(self, prompt: str) -> str:
        return self.query_with_usage(prompt)["response"]

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        try:
            msg = self.client.messages.create(
                model=self.model,
                temperature=self.temperature,
                max_tokens=self.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )

            usage = {
                "prompt_tokens": msg.usage.input_tokens,
                "completion_tokens": msg.usage.output_tokens,
                "total_tokens": msg.usage.input_tokens + msg.usage.output_tokens,
            }

            pricing = self.get_model_pricing()
            cost = (usage["prompt_tokens"] * pricing["input"] +
                    usage["completion_tokens"] * pricing["output"]) / 1_000_000

            return {
                "response": msg.content[0].text,
                "usage": usage,
                "cost": cost,
            }

        except Exception as e:
            return {
                "response": f"Error querying Anthropic: {str(e)}",
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": 0.0,
            }

    def get_name(self) -> str:
        return f"Anthropic({self.model})"

    def get_model_pricing(self) -> Dict[str, float]:
        return {"input": 3.0, "output": 15.0}
