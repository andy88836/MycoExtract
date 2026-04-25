"""
LLM Client Factory

Centralized factory for creating and managing multiple LLM clients.
Implements the "Heterogeneous Model, Smart Dispatching" strategy.

This factory:
1. Reads the extraction_config.yaml file
2. Creates specialized LLM clients for different tasks
3. Returns a dictionary of named clients for dependency injection
"""

import logging
import yaml
from pathlib import Path
from typing import Dict, Any

from src.llm_clients.providers import build_client, BaseLLMClient

logger = logging.getLogger(__name__)


class ClientFactory:
    """
    Factory for creating and managing multiple LLM clients.
    
    Supports heterogeneous model architecture where different models
    are used for different tasks (text vs multimodal extraction).
    """
    
    def __init__(self, config_path: str = "config/extraction_config.yaml"):
        """
        Initialize the client factory.
        
        Args:
            config_path: Path to the YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.clients: Dict[str, BaseLLMClient] = {}
        
        logger.info(f"📋 ClientFactory initialized with config: {config_path}")
    
    def _load_config(self) -> Dict[str, Any]:
        """
        Load configuration from YAML file.
        
        Returns:
            Configuration dictionary
            
        Raises:
            FileNotFoundError: If config file doesn't exist
            yaml.YAMLError: If config file is invalid
        """
        try:
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            logger.info(f"✓ Loaded configuration from: {self.config_path}")
            return config
        except FileNotFoundError:
            logger.error(f"❌ Configuration file not found: {self.config_path}")
            raise
        except yaml.YAMLError as e:
            logger.error(f"❌ Invalid YAML configuration: {e}")
            raise
    
    def build_all(self) -> Dict[str, BaseLLMClient]:
        """
        Build all LLM clients defined in the configuration.
        
        Returns:
            Dictionary mapping client names to client instances
            Example: {"text_client": <DeepSeekClient>, "multimodal_client": <ZhipuAIClient>}
        """
        llm_clients_config = self.config.get("llm_clients", {})
        
        if not llm_clients_config:
            logger.warning("⚠️  No LLM clients defined in configuration")
            return {}
        
        logger.info("🔧 Building LLM clients...")
        
        for client_name, client_config in llm_clients_config.items():
            try:
                provider = client_config.get("provider")
                model_name = client_config.get("model_name")
                
                if not provider or not model_name:
                    logger.error(f"❌ Missing provider or model_name for client: {client_name}")
                    continue
                
                logger.info(f"  Building {client_name}: {provider}/{model_name}")
                
                # Build the client using the existing build_client function
                client = build_client(provider, model_name)
                self.clients[client_name] = client
                
                logger.info(f"  ✓ Successfully created {client_name}")
                
            except Exception as e:
                logger.error(f"  ❌ Failed to create {client_name}: {e}")
                raise
        
        logger.info(f"✅ Built {len(self.clients)} LLM clients: {list(self.clients.keys())}")
        return self.clients
    
    def get_client(self, client_name: str) -> BaseLLMClient:
        """
        Get a specific client by name.
        
        Args:
            client_name: Name of the client (e.g., "text_client", "multimodal_client")
            
        Returns:
            LLM client instance
            
        Raises:
            KeyError: If client name doesn't exist
        """
        if client_name not in self.clients:
            raise KeyError(f"Client '{client_name}' not found. Available: {list(self.clients.keys())}")
        
        return self.clients[client_name]
    
    def get_config(self) -> Dict[str, Any]:
        """
        Get the full configuration dictionary.
        
        Returns:
            Complete configuration
        """
        return self.config
    
    def get_extraction_parameters(self) -> Dict[str, Any]:
        """
        Get extraction parameters from config.
        
        Returns:
            Extraction parameters dictionary
        """
        return self.config.get("extraction_parameters", {})
    
    def get_file_paths(self) -> Dict[str, str]:
        """
        Get file paths from config.
        
        Returns:
            File paths dictionary
        """
        return self.config.get("file_paths", {})
    
    def get_logging_config(self) -> Dict[str, Any]:
        """
        Get logging configuration.
        
        Returns:
            Logging configuration dictionary
        """
        return self.config.get("logging", {})


def create_clients_from_config(config_path: str = "config/extraction_config.yaml") -> Dict[str, BaseLLMClient]:
    """
    Convenience function to create all clients from a config file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        Dictionary of client name -> client instance
        
    Example:
        >>> clients = create_clients_from_config()
        >>> text_client = clients['text_client']
        >>> multimodal_client = clients['multimodal_client']
    """
    factory = ClientFactory(config_path)
    return factory.build_all()


if __name__ == "__main__":
    # Test the factory
    import sys
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    try:
        factory = ClientFactory()
        clients = factory.build_all()
        
        print(f"\n✅ Successfully created {len(clients)} clients:")
        for name, client in clients.items():
            print(f"  - {name}: {type(client).__name__}")
        
        print(f"\n📋 Configuration summary:")
        print(f"  Input dir: {factory.get_file_paths().get('input_dir')}")
        print(f"  Output JSON dir: {factory.get_file_paths().get('output_json_dir')}")
        print(f"  Temperature: {factory.get_extraction_parameters().get('temperature')}")
        
    except Exception as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
