import traceback
import random
import requests
from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Tuple
import re
from .models import CausalGraph
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


def _extract_text(resp) -> str:
    # 1) Responses API convenience
    t = getattr(resp, "output_text", None)
    if t:
        return t

    # 2) Responses API: walk output -> message -> content
    try:
        pieces = []
        for item in getattr(resp, "output", []) or []:
            if getattr(item, "type", None) == "message":
                for c in getattr(item, "content", []) or []:
                    ctype = getattr(c, "type", None) or (isinstance(c, dict) and c.get("type"))
                    if ctype in ("output_text", "text"):
                        # pydantic object or dict
                        text = getattr(c, "text", None) if hasattr(c, "text") else c.get("text")
                        if text:
                            pieces.append(text)
        if pieces:
            return "".join(pieces)
    except Exception:
        pass

    # 3) Chat Completions fallback (if you switch endpoints)
    try:
        return resp.choices[0].message.content
    except Exception:
        pass

    return str(resp)


def _extract_usage(resp):
    u = getattr(resp, "usage", None)
    if not u:
        return 0, 0, 0
    # Responses API names
    input_tokens = getattr(u, "input_tokens", getattr(u, "prompt_tokens", 0))
    output_tokens = getattr(u, "output_tokens", getattr(u, "completion_tokens", 0))
    total_tokens = getattr(u, "total_tokens", input_tokens + output_tokens)
    return input_tokens, output_tokens, total_tokens

class LLMInterface(ABC):
    """Abstract interface for LLM interaction."""
    
    @abstractmethod
    def query(self, prompt: str) -> str:
        """
        Query the LLM with a prompt and return response.
        
        Args:
            prompt: The prompt to send to the LLM
        
        Returns:
            The LLM's response as a string
        """
        pass
    
    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """Query the LLM and return response with usage stats."""
        # Default implementation for backward compatibility
        return {
            'response': self.query(prompt),
            'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
            'cost': 0.0
        }
    
    @abstractmethod
    def get_name(self) -> str:
        """Get the name/identifier of the LLM."""
        pass
    
    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for this model."""
        # Default pricing (can be overridden by subclasses)
        return {'input': 0.0, 'output': 0.0}
    
    def reset(self):
        """Reset any internal state (optional)."""
        pass





class LocalQwenLLM(LLMInterface):
    """Local Qwen model with API-style interface like OpenAI/Anthropic."""

    def __init__(
        self,
        model_path: str = "/opt/data/private/3d/qwen2.5-14B-merged/700",
        device: str = "cuda",
        temperature: float = 0.7,
        max_tokens: int = 4096,
        load_in_4bit: bool = True
    ):
        self.model_path = model_path
        self.device = device
        self.temperature = temperature
        self.max_tokens = max_tokens

        print(f"Loading LocalQwen model from: {model_path}")
        
        # 检查路径是否存在
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")

        try:
            # 强制离线模式，避免网络连接
            os.environ["TRANSFORMERS_OFFLINE"] = "1"
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            # 加载 tokenizer 和模型，设置离线模式
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_path,
                trust_remote_code=True,
                local_files_only=True  # 关键：只使用本地文件
            )
            
            # 确保 tokenizer 有 pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            self.model = AutoModelForCausalLM.from_pretrained(
                model_path,
                device_map="auto" if device == "cuda" else None,
                torch_dtype=torch.float16 if device == "cuda" else torch.float32,
                trust_remote_code=True,
                local_files_only=True  # 关键：只使用本地文件
            )
            
            print(f"Successfully loaded LocalQwen model from {model_path}")
            
        except Exception as e:
            print(f"Error loading model: {e}")
            traceback.print_exc()
            raise

    def query(self, prompt: str) -> str:
        """返回纯文本响应"""
        result = self.query_with_usage(prompt)
        return result["response"]

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """返回响应 + token 使用情况 + cost"""
        try:
            # 使用 Qwen 的聊天模板
            messages = [{"role": "user", "content": prompt}]
            
            # 应用聊天模板
            text = self.tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
            
            # 编码输入
            inputs = self.tokenizer(text, return_tensors="pt")
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}

            # 生成输出
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_tokens,
                    do_sample=True if self.temperature > 0 else False,
                    temperature=self.temperature,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id
                )

            # 解码响应（只取新生成的部分）
            response_ids = outputs[0][len(inputs["input_ids"][0]):]
            response_text = self.tokenizer.decode(response_ids, skip_special_tokens=True)

            # 计算 token 使用
            prompt_tokens = len(inputs["input_ids"][0])
            total_tokens = len(outputs[0])
            completion_tokens = total_tokens - prompt_tokens

            usage_data = {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens
            }

            return {
                "response": response_text.strip(),
                "usage": usage_data,
                "cost": 0.0  # 本地模型不收费
            }

        except Exception as e:
            error_msg = f"Error querying Local Qwen: {str(e)}"
            print(error_msg)
            traceback.print_exc()
            return {
                "response": error_msg,
                "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0},
                "cost": 0.0
            }

    def get_name(self) -> str:
        return f"LocalQwen({os.path.basename(self.model_path)})"

    def get_model_pricing(self) -> Dict[str, float]:
        return {"input": 0.0, "output": 0.0}



class DeepSeekLLM(LLMInterface):
    """
    DeepSeek API interface using official API.

    Requires requests package and DeepSeek API key.
    """

    def __init__(
            self,
            model: str = "deepseek-chat",
            api_key: Optional[str] = None,
            temperature: float = 0.7,
            max_tokens: int = 4096,
            base_url: str = "https://api.deepseek.com/v1"
    ):
        """
        Initialize DeepSeek LLM interface.

        Args:
            model: DeepSeek model to use
            api_key: DeepSeek API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            base_url: DeepSeek API base URL
        """
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
        """Query DeepSeek API."""
        result = self.query_with_usage(prompt)
        return result['response']

    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """Query DeepSeek API with usage tracking."""
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens,
                "stream": False
            }

            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            result = response.json()

            # Extract usage information
            usage = result.get('usage', {})
            usage_data = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }

            # Calculate cost based on model pricing
            pricing = self.get_model_pricing()
            cost = (usage_data['prompt_tokens'] * pricing['input'] +
                    usage_data['completion_tokens'] * pricing['output']) / 1_000_000

            return {
                'response': result['choices'][0]['message']['content'],
                'usage': usage_data,
                'cost': cost
            }

        except requests.exceptions.RequestException as e:
            return {
                'response': f"Error querying DeepSeek: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }
        except (KeyError, IndexError) as e:
            return {
                'response': f"Error parsing DeepSeek response: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }

    def get_name(self) -> str:
        """Get the model name."""
        return f"DeepSeek({self.model})"

    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for DeepSeek models."""
        # Pricing in dollars per 1M tokens (DeepSeek 官方定价)
        pricing_map = {
            'deepseek-chat': {'input': 0.14, 'output': 0.28},
            'deepseek-coder': {'input': 0.14, 'output': 0.28},
            'deepseek-reasoner': {'input': 1.1, 'output': 4.4},
        }
        return pricing_map.get(self.model, {'input': 0.14, 'output': 0.28})


class OpenRouterLLM(LLMInterface):
    """
    OpenRouter API interface for various LLM models.
    
    OpenRouter provides access to multiple models through a single API.
    """
    
    def __init__(
        self,
        model: str = "anthropic/claude-3.5-sonnet",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 40960,
        base_url: str = "https://openrouter.ai/api/v1"
    ):
        """
        Initialize OpenRouter LLM interface.
        
        Args:
            model: Model identifier (e.g., "anthropic/claude-3.5-sonnet", "openai/gpt-4")
            api_key: OpenRouter API key
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
            base_url: OpenRouter API base URL
        """
        if not api_key:
            raise ValueError("OpenRouter API key is required")
        
        self.model = model
        self.api_key = api_key
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.base_url = base_url
        self.headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
            # "HTTP-Referer":"http://localhost:3000",  # Required by OpenRouter
            # "X-Title": "Cre ativity Benchmark"  # Optional, for OpenRouter dashboard
        }
    
    def query(self, prompt: str) -> str:
        """Query OpenRouter API."""
        result = self.query_with_usage(prompt)
        return result['response']
    
    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """Query OpenRouter API with usage tracking."""
        try:
            url = f"{self.base_url}/chat/completions"
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": self.temperature,
                "max_tokens": self.max_tokens
            }
            
            response = requests.post(url, headers=self.headers, json=payload)
            response.raise_for_status()
            # print('result0',response)
            result = response.json()
            # print('result1',result['choices'][0]['message']['content'])
            # print('result2',result)
            # Extract usage information
            usage = result.get('usage', {})
            usage_data = {
                'prompt_tokens': usage.get('prompt_tokens', 0),
                'completion_tokens': usage.get('completion_tokens', 0),
                'total_tokens': usage.get('total_tokens', 0)
            }
            
            # Calculate cost based on model pricing
            pricing = self.get_model_pricing()
            cost = (usage_data['prompt_tokens'] * pricing['input'] + 
                   usage_data['completion_tokens'] * pricing['output']) / 1_000_000
            
            return {
                'response': result['choices'][0]['message']['content'],
                'usage': usage_data,
                'cost': cost
            }
            
        except requests.exceptions.RequestException as e:
            return {
                'response': f"Error querying OpenRouter: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }
        except (KeyError, IndexError) as e:
            return {
                'response': f"Error parsing OpenRouter response: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }
    
    def get_name(self) -> str:
        """Get the model name."""
        return f"OpenRouter({self.model})"
    
    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for common models."""
        # Pricing in dollars per 1M tokens
        pricing_map = {
            'anthropic/claude-3.5-sonnet': {'input': 3.0, 'output': 15.0},
            'anthropic/claude-3-opus': {'input': 15.0, 'output': 75.0},
            'openai/gpt-4o': {'input': 2.5, 'output': 10.0},
            'meta-llama/llama-3.3-70b-instruct': {'input': 0.038, 'output': 0.12},
            'google/gemini-2.5-pro': {'input': 1.25, 'output': 10.0},
            'deepseek/deepseek-chat': {'input': 0.4, 'output': 2.0},  # 添加这一行
            'deepseek/deepseek-coder': {'input': 0.4, 'output': 2.0},  # 添加这一行
            'deepseek/deepseek-r1': {'input': 0.4, 'output': 2.0},
        }
        return pricing_map.get(self.model, {'input': 1.0, 'output': 1.0})


class OpenAILLM(LLMInterface):
    """
    OpenAI API interface for GPT models.
    
    Requires openai package and API key.
    """
    
    def __init__(
        self, 
        model: str = "gpt-4",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 40960
    ):
        """
        Initialize OpenAI LLM interface.
        
        Args:
            model: OpenAI model to use
            api_key: OpenAI API key (uses environment variable if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        try:
            import openai
        except ImportError:
            raise ImportError("Please install openai package: pip install openai")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if not api_key:
            import os
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OpenAI API key must be provided or set as OPENAI_API_KEY environment variable")
        
        self.client = openai.OpenAI(api_key=api_key)
    
    def query(self, prompt: str) -> str:
        """Query OpenAI API."""
        result = self.query_with_usage(prompt)
        return result['response']
    
    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        try:
            # print(self.max_tokens)
            resp = self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": "You are an expert in causal inference and graph theory."},
                    {"role": "user", "content": prompt},
                ],
                reasoning={"effort": "medium"},
                max_output_tokens=self.max_tokens
            )

            text = _extract_text(resp)
            in_tok, out_tok, tot_tok = _extract_usage(resp)

            pricing = self.get_model_pricing()
            cost = (in_tok * pricing['input'] + out_tok * pricing['output']) / 1_000_000

            # print(text)
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
        """Get the model name."""
        return f"OpenAI({self.model})"
    
    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for OpenAI models."""
        # Pricing in dollars per 1M tokens
        pricing_map = {
            'gpt-4o': {'input': 2.5, 'output': 10.0},
            'gpt-4o-mini': {'input': 0.15, 'output': 0.6},
            'gpt-5': {'input': 1.25, 'output': 10.0}
        }
        return pricing_map.get(self.model, {'input': 10.0, 'output': 30.0})


class AnthropicLLM(LLMInterface):
    """
    Anthropic Claude API interface.
    
    Requires anthropic package and API key.
    """
    
    def __init__(
        self,
        model: str = "claude-3-opus-20240229",
        api_key: Optional[str] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ):
        """
        Initialize Anthropic LLM interface.
        
        Args:
            model: Anthropic model to use
            api_key: Anthropic API key (uses environment variable if not provided)
            temperature: Sampling temperature
            max_tokens: Maximum tokens in response
        """
        try:
            import anthropic
        except ImportError:
            raise ImportError("Please install anthropic package: pip install anthropic")
        
        self.model = model
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        self.client = anthropic.Anthropic(api_key=api_key)
    
    def query(self, prompt: str) -> str:
        """Query Anthropic API."""
        result = self.query_with_usage(prompt)
        return result['response']
    
    def query_with_usage(self, prompt: str) -> Dict[str, Any]:
        """Query Anthropic API with usage tracking."""
        try:
            response = self.client.messages.create(
                model=self.model,
                # max_tokens=self.max_tokens,
                temperature=self.temperature,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Extract usage information
            usage = {
                'prompt_tokens': response.usage.input_tokens if hasattr(response, 'usage') else 0,
                'completion_tokens': response.usage.output_tokens if hasattr(response, 'usage') else 0,
                'total_tokens': (response.usage.input_tokens + response.usage.output_tokens) if hasattr(response, 'usage') else 0
            }
            
            # Calculate cost
            pricing = self.get_model_pricing()
            cost = (usage['prompt_tokens'] * pricing['input'] + 
                   usage['completion_tokens'] * pricing['output']) / 1_000_000
            
            return {
                'response': response.content[0].text,
                'usage': usage,
                'cost': cost
            }
        except Exception as e:
            return {
                'response': f"Error querying Anthropic: {str(e)}",
                'usage': {'prompt_tokens': 0, 'completion_tokens': 0, 'total_tokens': 0},
                'cost': 0.0
            }
    
    def get_name(self) -> str:
        """Get the model name."""
        return f"Anthropic({self.model})"
    
    def get_model_pricing(self) -> Dict[str, float]:
        """Get pricing per 1M tokens for Anthropic models."""
        # Pricing in dollars per 1M tokens
        pricing_map = {
            'claude-3-opus-20240229': {'input': 15.0, 'output': 75.0},
            'claude-3-sonnet-20240229': {'input': 3.0, 'output': 15.0},
            'claude-3-haiku-20240307': {'input': 0.25, 'output': 1.25},
            'claude-3.5-sonnet-20241022': {'input': 3.0, 'output': 15.0},
        }
        return pricing_map.get(self.model, {'input': 3.0, 'output': 15.0})


class ResponseParser:
    """Parser for extracting causal graphs from LLM responses."""
    
    @staticmethod
    def parse_response(response: str) -> Optional[CausalGraph]:
        """
        Parse LLM response to extract causal graph.
        
        Handles various response formats and edge notations.
        
        Args:
            response: The LLM's response text
        
        Returns:
            CausalGraph if successfully parsed, None otherwise
        """
        try:
            # Extract nodes
            nodes = ResponseParser._extract_nodes(response)
            if not nodes:
                return None
            
            # Extract edges
            edges = ResponseParser._extract_edges(response)
            if not edges:
                # Try alternative extraction methods
                edges = ResponseParser._extract_edges_alternative(response)
            
            if nodes and edges:
                # Validate that edge nodes are in the node list
                edge_nodes = set()
                for src, dst in edges:
                    edge_nodes.add(src)
                    edge_nodes.add(dst)
                
                # Add any missing nodes
                for node in edge_nodes:
                    if node not in nodes:
                        nodes.append(node)
                
                return CausalGraph(nodes=sorted(nodes), edges=edges)
            
        except Exception as e:
            print(f"Error parsing response: {e}")
        
        return None
    
    @staticmethod
    def _extract_nodes(response: str) -> Optional[List[str]]:
        """Extract node list from response."""
        # Try different patterns
        patterns = [
            r'nodes?\s*\[([^\]]+)\]',
            r'nodes?\s*:\s*\[([^\]]+)\]',
            r'nodes?\s+(?:are\s+)?(\w+(?:,\s*\w+)*)',
            r'variables?\s*\[([^\]]+)\]',
            r'variables?\s+(?:are\s+)?(\w+(?:,\s*\w+)*)'
        ]
        
        for pattern in patterns:
            match = re.search(pattern, response, re.IGNORECASE)
            if match:
                nodes_str = match.group(1)
                # Clean and split
                nodes = [n.strip().strip("'\"") for n in nodes_str.split(',')]
                return [n for n in nodes if n]  # Filter empty strings
        
        return None
    
    @staticmethod
    def _extract_edges(response: str) -> List[tuple]:
        """Extract edges from response."""
        edges = []
        
        # Edge patterns to look for
        edge_patterns = [
            r'(\w+)\s*->\s*(\w+)',
            r'(\w+)\s*→\s*(\w+)',
            r'(\w+)\s+causes?\s+(\w+)',
            r'(\w+)\s+affects?\s+(\w+)',
            r'(\w+)\s+influences?\s+(\w+)'
        ]
        
        for pattern in edge_patterns:
            matches = re.findall(pattern, response, re.IGNORECASE)
            for match in matches:
                src, dst = match[0].strip(), match[1].strip()
                if src and dst and src != dst:  # Avoid self-loops
                    edges.append((src, dst))
        
        # Remove duplicates while preserving order
        seen = set()
        unique_edges = []
        for edge in edges:
            if edge not in seen:
                seen.add(edge)
                unique_edges.append(edge)
        
        return unique_edges
    
    @staticmethod
    def _extract_edges_alternative(response: str) -> List[tuple]:
        """Alternative method for extracting edges."""
        edges = []
        
        # Look for edges in format "edges: A->B, C->D, ..."
        edges_match = re.search(r'edges?:?\s*([^.]+)', response, re.IGNORECASE)
        if edges_match:
            edges_str = edges_match.group(1)
            
            # Split by comma and parse each edge
            edge_parts = edges_str.split(',')
            for part in edge_parts:
                # Try to extract edge from each part
                edge_match = re.search(r'(\w+)\s*(?:->|→)\s*(\w+)', part)
                if edge_match:
                    edges.append((edge_match.group(1), edge_match.group(2)))
        
        return edges