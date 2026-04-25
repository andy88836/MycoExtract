"""
Multi-Provider LLM Client Factory

This module implements a flexible LLM client management system supporting
multiple providers with both text-only and multimodal (text + image) capabilities.

Supported Providers:
- OpenAI (GPT-4o, GPT-4o-mini with Vision)
- DeepSeek (DeepSeek-Chat, OpenAI-compatible API)
- ZhipuAI (GLM-4.5v with Vision and Thinking mode)

All API keys and base URLs are loaded from environment variables.
"""

import os
import base64
import logging
from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
from pathlib import Path

import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class BaseLLMClient(ABC):
    """
    Abstract base class for LLM clients.
    
    All concrete client implementations must inherit from this class and
    implement the chat() method.
    """
    
    def __init__(self, model_name: str, api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize the LLM client.
        
        Args:
            model_name: Name of the model to use
            api_key: API key for authentication (if required)
            base_url: Base URL for the API endpoint (if custom)
        """
        self.model_name = model_name
        self.api_key = api_key
        self.base_url = base_url
        
    @abstractmethod
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, **kwargs) -> str:
        """
        Send a chat request to the LLM.
        
        Args:
            messages: List of message dictionaries with 'role' and 'content' keys
            is_multimodal: Whether the request includes images
            **kwargs: Additional provider-specific parameters
            
        Returns:
            String response from the LLM
            
        Raises:
            Exception: If the API request fails
        """
        pass
    
    # 静态方法不能访问或修改实例的状态 它也不能访问或修改类的状态
    '''
    功能独立：这两个函数的功能——将图片文件编码为 Base64 字符串，
    以及根据文件扩展名确定 MIME 类型——是完全自包含的。它们的执行结果仅依赖于传入的 image_path 参数
    而完全不依赖于 BaseLLMClient 类的任何特定实例（如 model_name、api_key 或 base_url）。
    '''

    @staticmethod
    def encode_image(image_path: str) -> str:
        """
        Encode an image file to base64 string.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Base64-encoded image string
        """
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')
    
    @staticmethod
    def get_image_mime_type(image_path: str) -> str:
        """
        Determine the MIME type of an image file.
        根据一个图片文件的路径,判断出它的文件类型,并返回一个标准的MIME类型字符串
        
        Args:
            image_path: Path to the image file
            
        Returns:
            MIME type string (e.g., 'image/jpeg', 'image/png')
        """
        extension = Path(image_path).suffix.lower()
        mime_types = {
            '.jpg': 'image/jpeg',
            '.jpeg': 'image/jpeg',
            '.png': 'image/png',
            '.gif': 'image/gif',
            '.webp': 'image/webp'
        }
        return mime_types.get(extension, 'image/jpeg')


class OpenAIClient(BaseLLMClient):
    """
    OpenAI API client supporting both text and vision models.
    
    Supports models like GPT-4, GPT-4 Turbo, and GPT-4 Vision.
    """
    
    def __init__(self, model_name: str = "gpt-5", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize OpenAI client.
        
        Args:
            model_name: OpenAI model name (default: gpt-5)
            api_key: OpenAI API key (from OPENAI_API_KEY env var if not provided)
            base_url: Custom base URL (optional)
        """
        api_key = api_key or os.getenv("OPENAI_API_KEY")
        base_url = base_url or os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
        
        if not api_key:
            raise ValueError("OpenAI API key not found. Set OPENAI_API_KEY environment variable.")
        
        super().__init__(model_name, api_key, base_url)
        logger.info(f"Initialized OpenAI client with model: {model_name}")
    
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, **kwargs) -> str:
        """
        Send a chat completion request to OpenAI.
        
        Args:
            messages: List of message dictionaries
            is_multimodal: Whether images are included
            **kwargs: Additional parameters (temperature, max_tokens, etc.)
            
        Returns:
            Model response text
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        # Process messages for multimodal content
        processed_messages = self._process_messages(messages, is_multimodal)
        
        payload = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": kwargs.get("temperature", 0.1),
            "max_tokens": kwargs.get("max_tokens", 4096)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", 120)
            )
            response.raise_for_status()
            
            result = response.json()
            return result["choices"][0]["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"OpenAI API request failed: {e}")
            raise
    
    def _process_messages(self, messages: List[Dict[str, Any]], is_multimodal: bool) -> List[Dict[str, Any]]:
        """
        Process messages to handle multimodal content.
        
        Args:
            messages: Raw message list
            is_multimodal: Whether to process images
            
        Returns:
            Processed message list in OpenAI format
        """
        if not is_multimodal:
            return messages
        
        processed = []
        for msg in messages:
            if isinstance(msg.get("content"), list):
                # Already in multimodal format
                processed.append(msg)
            elif "image_path" in msg:
                # Convert image path to multimodal format
                content = [
                    {"type": "text", "text": msg.get("text", "")}
                ]
                
                # Handle single or multiple images
                image_paths = msg["image_path"] if isinstance(msg["image_path"], list) else [msg["image_path"]]
                
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        base64_image = self.encode_image(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                
                processed.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })
            else:
                processed.append(msg)
        
        return processed


class AnthropicClient(BaseLLMClient):
    """
    Anthropic Claude API client supporting vision models.
    
    Supports Claude 3 models (Opus, Sonnet, Haiku).
    """
    
    def __init__(self, model_name: str = "claude-3-5-sonnet-20241022", api_key: Optional[str] = None):
        """
        Initialize Anthropic client.
        
        Args:
            model_name: Claude model name
            api_key: Anthropic API key (from ANTHROPIC_API_KEY env var if not provided)
        """
        api_key = api_key or os.getenv("ANTHROPIC_API_KEY")
        
        if not api_key:
            raise ValueError("Anthropic API key not found. Set ANTHROPIC_API_KEY environment variable.")
        
        super().__init__(model_name, api_key, "https://api.anthropic.com")
        logger.info(f"Initialized Anthropic client with model: {model_name}")
    
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, **kwargs) -> str:
        """
        Send a message to Claude.
        
        Args:
            messages: List of message dictionaries
            is_multimodal: Whether images are included
            **kwargs: Additional parameters
            
        Returns:
            Model response text
        """
        headers = {
            "x-api-key": self.api_key,
            "anthropic-version": "2023-06-01",
            "Content-Type": "application/json"
        }
        
        # Process messages for Anthropic format
        processed_messages = self._process_messages(messages, is_multimodal)
        
        payload = {
            "model": self.model_name,
            "messages": processed_messages,
            "max_tokens": kwargs.get("max_tokens", 4096),
            "temperature": kwargs.get("temperature", 0.1)
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/v1/messages",
                headers=headers,
                json=payload,
                timeout=kwargs.get("timeout", 120)
            )
            response.raise_for_status()
            
            result = response.json()
            return result["content"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Anthropic API request failed: {e}")
            raise
    
    def _process_messages(self, messages: List[Dict[str, Any]], is_multimodal: bool) -> List[Dict[str, Any]]:
        """
        Process messages for Anthropic's format.
        
        Args:
            messages: Raw message list
            is_multimodal: Whether to process images
            
        Returns:
            Processed message list in Anthropic format
        """
        if not is_multimodal:
            return messages
        
        processed = []
        for msg in messages:
            if "image_path" in msg:
                content = []
                
                # Add text if present
                if msg.get("text"):
                    content.append({
                        "type": "text",
                        "text": msg["text"]
                    })
                
                # Handle images
                image_paths = msg["image_path"] if isinstance(msg["image_path"], list) else [msg["image_path"]]
                
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        base64_image = self.encode_image(img_path)
                        mime_type = self.get_image_mime_type(img_path)
                        
                        content.append({
                            "type": "image",
                            "source": {
                                "type": "base64",
                                "media_type": mime_type,
                                "data": base64_image
                            }
                        })
                
                processed.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })
            else:
                processed.append(msg)
        
        return processed


class GeminiClient(BaseLLMClient):
    """
    Google Gemini API client supporting vision models.
    
    Supports Gemini Pro and Gemini Pro Vision models.
    """
    
    def __init__(self, model_name: str = "gemini-2.5-pro", api_key: Optional[str] = None):
        """
        Initialize Gemini client.
        
        Args:
            model_name: Gemini model name
            api_key: Google API key (from GOOGLE_API_KEY env var if not provided)
        """
        api_key = api_key or os.getenv("GOOGLE_API_KEY")
        
        if not api_key:
            raise ValueError("Google API key not found. Set GOOGLE_API_KEY environment variable.")
        
        super().__init__(model_name, api_key, "https://generativelanguage.googleapis.com")
        logger.info(f"Initialized Gemini client with model: {model_name}")
    
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, **kwargs) -> str:
        """
        Send a request to Gemini.
        
        Args:
            messages: List of message dictionaries
            is_multimodal: Whether images are included
            **kwargs: Additional parameters
            
        Returns:
            Model response text
        """
        # Convert messages to Gemini format
        contents = self._process_messages(messages, is_multimodal)
        
        payload = {
            "contents": contents,
            "generationConfig": {
                "temperature": kwargs.get("temperature", 0.1),
                "maxOutputTokens": kwargs.get("max_tokens", 4096)
            }
        }
        
        try:
            url = f"{self.base_url}/v1beta/models/{self.model_name}:generateContent?key={self.api_key}"
            
            response = requests.post(
                url,
                json=payload,
                timeout=kwargs.get("timeout", 120)
            )
            response.raise_for_status()
            
            result = response.json()
            return result["candidates"][0]["content"]["parts"][0]["text"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Gemini API request failed: {e}")
            raise
    
    def _process_messages(self, messages: List[Dict[str, Any]], is_multimodal: bool) -> List[Dict[str, Any]]:
        """
        Process messages for Gemini's format.
        
        Args:
            messages: Raw message list
            is_multimodal: Whether to process images
            
        Returns:
            Processed content list in Gemini format
        """
        contents = []
        
        for msg in messages:
            parts = []
            
            if "image_path" in msg and is_multimodal:
                # Add text first if present
                if msg.get("text"):
                    parts.append({"text": msg["text"]})
                
                # Add images
                image_paths = msg["image_path"] if isinstance(msg["image_path"], list) else [msg["image_path"]]
                
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        base64_image = self.encode_image(img_path)
                        mime_type = self.get_image_mime_type(img_path)
                        
                        parts.append({
                            "inline_data": {
                                "mime_type": mime_type,
                                "data": base64_image
                            }
                        })
            else:
                # Text only
                parts.append({"text": msg.get("content", msg.get("text", ""))})
            
            contents.append({
                "role": "user" if msg.get("role") == "user" else "model",
                "parts": parts
            })
        
        return contents


class OllamaClient(BaseLLMClient):
    """
    Ollama local LLM client supporting vision models.
    
    Supports local models like LLaVA, Bakllava, and other vision-enabled models.
    """
    
    def __init__(self, model_name: str = "llava", base_url: Optional[str] = None):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Ollama model name
            base_url: Ollama server URL (from OLLAMA_BASE_URL env var or default)
        """
        base_url = base_url or os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
        
        super().__init__(model_name, None, base_url)
        logger.info(f"Initialized Ollama client with model: {model_name} at {base_url}")
    
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, **kwargs) -> str:
        """
        Send a request to Ollama.
        
        Args:
            messages: List of message dictionaries
            is_multimodal: Whether images are included
            **kwargs: Additional parameters
            
        Returns:
            Model response text
        """
        processed_messages = self._process_messages(messages, is_multimodal)
        
        payload = {
            "model": self.model_name,
            "messages": processed_messages,
            "stream": False,
            "options": {
                "temperature": kwargs.get("temperature", 0.1)
            }
        }
        
        try:
            response = requests.post(
                f"{self.base_url}/api/chat",
                json=payload,
                timeout=kwargs.get("timeout", 180)
            )
            response.raise_for_status()
            
            result = response.json()
            return result["message"]["content"]
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Ollama API request failed: {e}")
            raise
    
    def _process_messages(self, messages: List[Dict[str, Any]], is_multimodal: bool) -> List[Dict[str, Any]]:
        """
        Process messages for Ollama's format.
        
        Args:
            messages: Raw message list
            is_multimodal: Whether to process images
            
        Returns:
            Processed message list in Ollama format
        """
        if not is_multimodal:
            return messages
        
        processed = []
        for msg in messages:
            if "image_path" in msg:
                # Ollama uses base64 images in the 'images' field
                images = []
                image_paths = msg["image_path"] if isinstance(msg["image_path"], list) else [msg["image_path"]]
                
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        images.append(self.encode_image(img_path))
                
                processed.append({
                    "role": msg.get("role", "user"),
                    "content": msg.get("text", ""),
                    "images": images
                })
            else:
                processed.append(msg)
        
        return processed


class DeepSeekClient(OpenAIClient):
    """
    DeepSeek API client (OpenAI-compatible).
    
    DeepSeek provides an OpenAI-compatible API, so we inherit from OpenAIClient
    and just change the base URL and authentication.
    """
    
    def __init__(self, model_name: str = "deepseek-chat", api_key: Optional[str] = None):
        """
        Initialize DeepSeek client.
        
        Args:
            model_name: DeepSeek model name
            api_key: DeepSeek API key (from DEEPSEEK_API_KEY env var if not provided)
        """
        api_key = api_key or os.getenv("DEEPSEEK_API_KEY")
        base_url = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        
        if not api_key:
            raise ValueError("DeepSeek API key not found. Set DEEPSEEK_API_KEY environment variable.")
        
        # Initialize with OpenAI-compatible parameters
        super(OpenAIClient, self).__init__(model_name, api_key, base_url)
        logger.info(f"Initialized DeepSeek client with model: {model_name}")


class LongCatClient(OpenAIClient):
    """
    LongCat API client (OpenAI-compatible).
    
    LongCat provides an OpenAI-compatible API for long-context models.
    """
    
    def __init__(self, model_name: str = "longcat-70b", api_key: Optional[str] = None, base_url: Optional[str] = None):
        """
        Initialize LongCat client.
        
        Args:
            model_name: LongCat model name
            api_key: LongCat API key (from LONGCAT_API_KEY env var if not provided)
            base_url: LongCat base URL (from LONGCAT_BASE_URL env var if not provided)
        """
        api_key = api_key or os.getenv("LONGCAT_API_KEY")
        base_url = base_url or os.getenv("LONGCAT_BASE_URL", "https://api.longcat.ai/v1")
        
        if not api_key:
            raise ValueError("LongCat API key not found. Set LONGCAT_API_KEY environment variable.")
        
        # Initialize with OpenAI-compatible parameters
        super(OpenAIClient, self).__init__(model_name, api_key, base_url)
        logger.info(f"Initialized LongCat client with model: {model_name}")


class ZhipuAIClient(BaseLLMClient):
    """
    ZhipuAI (智谱AI) API client supporting GLM models.
    
    Supports GLM-4.5, GLM-4V (vision), and other ZhipuAI models.
    Uses the official zhipuai Python SDK.
    """
    
    def __init__(self, model_name: str = "glm-4.6", api_key: Optional[str] = None):
        """
        Initialize ZhipuAI client.
        
        Args:
            model_name: ZhipuAI model name (glm-4.5, glm-4-flash, glm-4v, etc.)
            api_key: ZhipuAI API key (from ZHIPUAI_API_KEY env var if not provided)
        """
        api_key = api_key or os.getenv("ZHIPUAI_API_KEY")
        
        if not api_key:
            raise ValueError("ZhipuAI API key not found. Set ZHIPUAI_API_KEY environment variable.")
        
        super().__init__(model_name, api_key, "https://open.bigmodel.cn/api/paas/v4")
        
        # Import ZhipuAI SDK
        try:
            from zai import ZhipuAiClient
            self.client = ZhipuAiClient(api_key=api_key)
        except ImportError:
            raise ImportError(
                "ZhipuAI SDK not found. Install it with: pip install zai"
            )
        
        logger.info(f"Initialized ZhipuAI client with model: {model_name}")
    
    def chat(self, messages: List[Dict[str, Any]], is_multimodal: bool = False, **kwargs) -> str:
        """
        Send a chat request to ZhipuAI.
        
        Args:
            messages: List of message dictionaries
            is_multimodal: Whether images are included
            **kwargs: Additional parameters (temperature, max_tokens, thinking, etc.)
            
        Returns:
            Model response text
        """
        # Process messages for multimodal content
        processed_messages = self._process_messages(messages, is_multimodal)
        
        # Build request parameters
        request_params = {
            "model": self.model_name,
            "messages": processed_messages,
            "temperature": kwargs.get("temperature", 0.6),
            "max_tokens": kwargs.get("max_tokens", 4096)
        }
        
        # Add thinking mode if specified (for models that support it)
        if kwargs.get("thinking"):
            request_params["thinking"] = kwargs["thinking"]
        elif "glm-4" in self.model_name.lower():
            # Enable thinking mode by default for GLM-4 models
            request_params["thinking"] = {"type": "enabled"}
        
        try:
            response = self.client.chat.completions.create(**request_params)
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"ZhipuAI API request failed: {e}")
            raise
    
    def _process_messages(self, messages: List[Dict[str, Any]], is_multimodal: bool) -> List[Dict[str, Any]]:
        """
        Process messages for ZhipuAI's format.
        
        Args:
            messages: Raw message list
            is_multimodal: Whether to process images
            
        Returns:
            Processed message list in ZhipuAI format
        """
        if not is_multimodal:
            return messages
        
        processed = []
        for msg in messages:
            if "image_path" in msg:
                # ZhipuAI vision format (similar to OpenAI)
                content = []
                
                # Add text if present
                if msg.get("text"):
                    content.append({
                        "type": "text",
                        "text": msg["text"]
                    })
                
                # Handle single or multiple images
                image_paths = msg["image_path"] if isinstance(msg["image_path"], list) else [msg["image_path"]]
                
                for img_path in image_paths:
                    if os.path.exists(img_path):
                        base64_image = self.encode_image(img_path)
                        content.append({
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        })
                
                processed.append({
                    "role": msg.get("role", "user"),
                    "content": content
                })
            else:
                processed.append(msg)
        
        return processed


def build_client(provider: str, model_name: Optional[str] = None) -> BaseLLMClient:
    """
    Factory function to build an LLM client based on provider name.
    
    Args:
        provider: Provider name (openai, anthropic, gemini, ollama, deepseek, longcat, zhipuai)
        model_name: Optional model name (uses provider default if not specified)
        
    Returns:
        Initialized LLM client instance
        
    Raises:
        ValueError: If provider is not supported
        
    Examples:
        >>> client = build_client("openai", "gpt-5")
        >>> client = build_client("anthropic")  # Uses default model
        >>> client = build_client("ollama", "llava")
        >>> client = build_client("zhipuai", "glm-4.5")
    """
    provider = provider.lower().strip()
    
    provider_map = {
        "openai": (OpenAIClient, "gpt-5"),
        "anthropic": (AnthropicClient, "claude-3-5-sonnet-20241022"),
        "gemini": (GeminiClient, "gemini-2.5-pro"),
        "ollama": (OllamaClient, "llava"),
        "deepseek": (DeepSeekClient, "deepseek-chat"),
        "longcat": (LongCatClient, "longcat-70b"),
        "zhipuai": (ZhipuAIClient, "glm-4.6")
    }
    
    if provider not in provider_map:
        raise ValueError(
            f"Unsupported provider: {provider}. "
            f"Supported providers: {', '.join(provider_map.keys())}"
        )
    
    client_class, default_model = provider_map[provider]
    model = model_name or default_model
    
    logger.info(f"Building {provider} client with model: {model}")
    return client_class(model_name=model)


if __name__ == "__main__":
    """
    Test the LLM client factory and providers.
    """
    logging.basicConfig(level=logging.INFO)
    
    # Test building clients
    providers = ["openai", "anthropic", "gemini", "ollama", "deepseek", "longcat", "zhipuai"]
    
    for provider in providers:
        try:
            client = build_client(provider)
            print(f"✓ Successfully built {provider} client: {client.__class__.__name__}")
        except ValueError as e:
            print(f"✗ Failed to build {provider} client: {e}")