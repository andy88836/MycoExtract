"""
Multi-Provider LLM Client Factory

This module implements a flexible, extensible LLM client management system supporting
multiple providers with both text-only and multimodal (text + image) capabilities.

Supported Providers:
- OpenAI (GPT-4, GPT-4 Vision)
- Anthropic (Claude 3 models)
- Google Gemini (Gemini Pro Vision)
- Ollama (Local models with vision support)
- DeepSeek (OpenAI-compatible API)
- LongCat (OpenAI-compatible API)
- ZhipuAI (GLM-4, GLM-4V with thinking mode)
- Moonshot/Kimi (Kimi Vision models)

All API keys and base URLs are loaded from environment variables.
"""

import os
import base64
import json
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url

    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, json_mode: bool = False, **kwargs) -> str:
        """
        Send a chat request to the LLM.
        
        Args:
            messages: List of message dicts (role, content/text, image_path)
            is_multimodal: Whether the request involves images
            json_mode: Whether to enforce JSON output
            **kwargs: Additional arguments (temperature, max_tokens, etc.)
            
        Returns:
            The response text
        """
        pass

    def encode_image(self, image_path: str) -> str:
        """Encode image to base64."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')


class OpenAIClient(BaseLLMClient):
    """Client for OpenAI and compatible APIs."""
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        super().__init__(model_name, api_key, base_url)
        from openai import OpenAI
        
        self.client = OpenAI(
            api_key=api_key or os.getenv("OPENAI_API_KEY"),
            base_url=base_url or os.getenv("OPENAI_BASE_URL")
        )
        
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, json_mode: bool = False, **kwargs) -> str:
        processed_messages = self._process_messages(messages, is_multimodal)
        
        params = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4096)
        }
        
        # Support for extra_body (needed for MiMo/Xiaomi thinking parameter)
        if "extra_body" in kwargs:
            params["extra_body"] = kwargs["extra_body"]
        
        if json_mode:
            params["response_format"] = {"type": "json_object"}
            
        try:
            response = self.client.chat.completions.create(**params)
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise

    def _process_messages(self, messages: List[Dict[str, Any]], is_multimodal: bool) -> List[Dict[str, Any]]:
        if not is_multimodal:
            # Standard text format
            return [{"role": m["role"], "content": m.get("content") or m.get("text")} for m in messages]
            
        processed = []
        for msg in messages:
            content = []
            if msg.get("text"):
                content.append({"type": "text", "text": msg["text"]})
            
            if "image_path" in msg:
                img_paths = msg["image_path"] if isinstance(msg["image_path"], list) else [msg["image_path"]]
                for path in img_paths:
                    if os.path.exists(path):
                        b64 = self.encode_image(path)
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                        })
            
            processed.append({"role": msg.get("role", "user"), "content": content})
        return processed


class GPT5Client(OpenAIClient):
    """GPT-5 Client (via Relay/ChatAnywhere)."""
    pass


class GPT4oClient(OpenAIClient):
    """GPT-4o Client."""
    pass


class DeepSeekClient(OpenAIClient):
    """DeepSeek Client (OpenAI Compatible)."""
    
    def __init__(self, model_name: str):
        super().__init__(
            model_name, 
            api_key=os.getenv("DEEPSEEK_API_KEY"),
            base_url="https://api.deepseek.com"
        )


class LongCatClient(OpenAIClient):
    """LongCat Client (OpenAI Compatible)."""
    
    def __init__(self, model_name: str):
        super().__init__(
            model_name, 
            api_key=os.getenv("LONGCAT_API_KEY"),
            base_url="https://api.longcat.com/v1"
        )


class MoonshotClient(OpenAIClient):
    """Moonshot (Kimi) Client with Vision Support."""
    
    def __init__(self, model_name: str):
        super().__init__(
            model_name, 
            api_key=os.getenv("MOONSHOT_API_KEY"),
            base_url="https://api.moonshot.cn/v1"
        )


class MiMoClient(OpenAIClient):
    """Xiaomi MiMo Client (OpenAI Compatible)."""

    def __init__(self, model_name: str):
        super().__init__(
            model_name,
            api_key=os.getenv("MIMO_API_KEY"),
            base_url="https://api.xiaomimimo.com/v1"
        )

    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, json_mode: bool = False, **kwargs) -> str:
        # Add MiMo specific defaults
        if "extra_body" not in kwargs:
             kwargs["extra_body"] = {}
        
        # Disable thinking by default as per user example, unless specified otherwise
        if "thinking" not in kwargs.get("extra_body", {}):
            kwargs["extra_body"]["thinking"] = {"type": "disabled"}

        return super().chat(messages, is_multimodal, json_mode, **kwargs)


class AnthropicClient(BaseLLMClient):
    """Client for Anthropic (Claude)."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        from anthropic import Anthropic
        self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))
        
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, json_mode: bool = False, **kwargs) -> str:
        processed_messages = self._process_messages(messages, is_multimodal)
        
        system_prompt = "You are a helpful assistant."
        if json_mode:
            system_prompt += " You must output valid JSON only."
            
        try:
            response = self.client.messages.create(
                model=self.model_name,
                messages=processed_messages,
                system=system_prompt,
                max_tokens=kwargs.get("max_tokens", 4096),
                temperature=kwargs.get("temperature", 0.1)
            )
            return response.content[0].text
        except Exception as e:
            logger.error(f"Anthropic API request failed: {e}")
            raise

    def _process_messages(self, messages: List[Dict[str, Any]], is_multimodal: bool) -> List[Dict[str, Any]]:
        processed = []
        for msg in messages:
            content = []
            if msg.get("text"):
                content.append({"type": "text", "text": msg["text"]})
                
            if is_multimodal and "image_path" in msg:
                img_paths = msg["image_path"] if isinstance(msg["image_path"], list) else [msg["image_path"]]
                for path in img_paths:
                    if os.path.exists(path):
                        b64 = self.encode_image(path)
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": "image/jpeg",
                                "data": b64
                            }
                        })
            processed.append({"role": msg.get("role", "user"), "content": content})
        return processed


class GeminiClient(BaseLLMClient):
    """Client for Google Gemini."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        import google.generativeai as genai
        genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))
        self.model = genai.GenerativeModel(model_name)
        
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, json_mode: bool = False, **kwargs) -> str:
        # Simplified Gemini implementation
        content = []
        for msg in messages:
            if msg.get("text"):
                content.append(msg["text"])
            if is_multimodal and "image_path" in msg:
                # Gemini handles images differently, simplified here
                pass
        
        generation_config = {
            "temperature": kwargs.get("temperature", 0.1),
            "max_output_tokens": kwargs.get("max_tokens", 4096)
        }
        
        if json_mode:
            generation_config["response_mime_type"] = "application/json"
            
        try:
            response = self.model.generate_content(content, generation_config=generation_config)
            return response.text
        except Exception as e:
            logger.error(f"Gemini API request failed: {e}")
            raise


class OllamaClient(BaseLLMClient):
    """Client for local Ollama models."""
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        self.base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, json_mode: bool = False, **kwargs) -> str:
        # Simplified Ollama implementation
        prompt = messages[-1].get("text", "")
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1),
                "num_predict": kwargs.get("max_tokens", 4096)
            }
        }
        
        if json_mode:
            payload["format"] = "json"
            
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload)
            response.raise_for_status()
            return response.json()["response"]
        except Exception as e:
            logger.error(f"Ollama API request failed: {e}")
            raise


class ZhipuAIClient(BaseLLMClient):
    """Client for ZhipuAI (GLM models) with token tracking."""
    
    # Class-level token counters for global tracking
    total_prompt_tokens = 0
    total_completion_tokens = 0
    total_requests = 0
    
    def __init__(self, model_name: str):
        super().__init__(model_name)
        from zhipuai import ZhipuAI
        self.client = ZhipuAI(api_key=os.getenv("ZHIPUAI_API_KEY"))
        # Instance-level counters
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.request_count = 0
        
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, json_mode: bool = False, **kwargs) -> str:
        processed_messages = self._process_messages(messages, is_multimodal)
        
        params = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4096)
        }
        
        # Thinking mode support
        if kwargs.get("thinking"):
            params["thinking"] = kwargs["thinking"]
        elif "glm-4" in self.model_name.lower() and kwargs.get("thinking") is not None:
             # Auto-enable for GLM-4 unless explicitly disabled
             pass
             
        # JSON mode support
        if json_mode:
            # Check if model supports it, GLM-4 usually does
            if "glm-4" in self.model_name.lower():
                params["response_format"] = {"type": "json_object"}

        # 设置超时时间（视觉模型需要更长超时）
        timeout = kwargs.get("timeout", 300 if is_multimodal else 120)

        try:
            response = self.client.chat.completions.create(
                **params,
                timeout=timeout  # 添加超时设置
            )
            
            # Track token usage
            if hasattr(response, 'usage') and response.usage:
                prompt_tokens = response.usage.prompt_tokens or 0
                completion_tokens = response.usage.completion_tokens or 0
                
                # Update instance counters
                self.prompt_tokens += prompt_tokens
                self.completion_tokens += completion_tokens
                self.request_count += 1
                
                # Update class-level counters
                ZhipuAIClient.total_prompt_tokens += prompt_tokens
                ZhipuAIClient.total_completion_tokens += completion_tokens
                ZhipuAIClient.total_requests += 1
                
                logger.debug(f"Token usage - Prompt: {prompt_tokens}, Completion: {completion_tokens}")
            
            return response.choices[0].message.content
        except Exception as e:
            logger.error(f"ZhipuAI API request failed: {e}")
            raise
    
    def get_token_stats(self) -> Dict[str, int]:
        """Get token statistics for this client instance."""
        return {
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "request_count": self.request_count
        }
    
    @classmethod
    def get_global_token_stats(cls) -> Dict[str, int]:
        """Get global token statistics across all ZhipuAI client instances."""
        return {
            "prompt_tokens": cls.total_prompt_tokens,
            "completion_tokens": cls.total_completion_tokens,
            "total_tokens": cls.total_prompt_tokens + cls.total_completion_tokens,
            "request_count": cls.total_requests
        }
    
    @classmethod
    def reset_global_stats(cls):
        """Reset global token statistics."""
        cls.total_prompt_tokens = 0
        cls.total_completion_tokens = 0
        cls.total_requests = 0

    def _process_messages(self, messages: List[Dict[str, Any]], is_multimodal: bool) -> List[Dict[str, Any]]:
        if not is_multimodal:
            return [{"role": m["role"], "content": m.get("content") or m.get("text")} for m in messages]
            
        processed = []
        for msg in messages:
            content = []
            if msg.get("text"):
                content.append({"type": "text", "text": msg["text"]})
                
            if "image_path" in msg:
                img_paths = msg["image_path"] if isinstance(msg["image_path"], list) else [msg["image_path"]]
                for path in img_paths:
                    if os.path.exists(path):
                        b64 = self.encode_image(path)
                        # Determine image type from extension
                        ext = os.path.splitext(path)[1].lower()
                        mime_type = {
                            ".jpg": "image/jpeg",
                            ".jpeg": "image/jpeg", 
                            ".png": "image/png",
                            ".gif": "image/gif",
                            ".webp": "image/webp"
                        }.get(ext, "image/jpeg")
                        # ZhipuAI requires data URI format
                        content.append({
                            "type": "image_url",
                            "image_url": {"url": f"data:{mime_type};base64,{b64}"}
                        })
                    else:
                        logger.warning(f"Image file not found: {path}")
            
            # 🔧 修复：确保 content 不为空
            if not content:
                logger.error(f"Empty content for message: {msg}")
                raise ValueError(f"Message must have either 'text' or valid 'image_path': {msg}")
            
            processed.append({"role": msg.get("role", "user"), "content": content})
        return processed


def build_client(provider: str, model_name: Optional[str] = None) -> BaseLLMClient:
    """
    Factory function to build an LLM client based on provider name.
    """
    provider = provider.lower().strip()
    
    provider_map = {
        "gpt5": (GPT5Client, "gpt-5"),
        "gpt-5": (GPT5Client, "gpt-5"),
        "openai": (OpenAIClient, "gpt-4o"),
        "gpt4o": (GPT4oClient, "gpt-4o"),
        "anthropic": (AnthropicClient, "claude-3-5-sonnet-20241022"),
        "gemini": (GeminiClient, "gemini-1.5-pro"),
        "ollama": (OllamaClient, "llava"),
        "deepseek": (DeepSeekClient, "deepseek-reasoner"),
        "longcat": (LongCatClient, "longcat-70b"),
        "zhipuai": (ZhipuAIClient, "glm-4.5-air"),
        "moonshot": (MoonshotClient, "moonshot-v1-8k-vision-preview"),
        "kimi": (MoonshotClient, "moonshot-v1-8k-vision-preview"),
        "mimo": (MiMoClient, "mimo-v2-flash"),
        "xiaomi": (MiMoClient, "mimo-v2-flash")
    }
    
    if provider not in provider_map:
        raise ValueError(f"Unsupported provider: {provider}")
    
    client_class, default_model = provider_map[provider]
    model = model_name or default_model
    
    logger.info(f"Building {provider} client with model: {model}")
    return client_class(model_name=model)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    try:
        client = build_client("openai")
        print(f"✓ Built client: {client.__class__.__name__}")
    except Exception as e:
        print(f"✗ Error: {e}")