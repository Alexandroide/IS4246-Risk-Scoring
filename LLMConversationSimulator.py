"""
LLM Conversation Simulator
Simulates conversations with different LLM models using scenarios from text files
and records responses for risk analysis.
"""

import os
import json
import yaml
import logging
import warnings
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import time

# Suppress common HuggingFace warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"
warnings.filterwarnings(
    "ignore", 
    message=".*A new version of the following files was downloaded.*", 
    module="transformers.utils.hub"
)
warnings.filterwarnings(
    "ignore", 
    message=".*past_key_values is passed but not used.*", 
    module="transformers.models.qwen2.modeling_qwen2"
)

# Configuration and Data Models
@dataclass
class ScenarioData:
    """Represents a conversation scenario."""
    scenario_id: str
    user: str
    system_prompt: str
    messages: List[str]
    metadata: Dict[str, Any]

@dataclass
class ConversationTurn:
    """Represents a single turn in a conversation."""
    role: str  # 'user' or 'assistant'
    content: str
    timestamp: str

@dataclass
class ConversationRecord:
    """Complete record of a simulated conversation."""
    scenario_id: str
    model_name: str
    provider: str
    timestamp: str
    user: str
    system_prompt: str
    conversation: List[Dict[str, str]]
    model_parameters: Dict[str, Any]
    risk_metrics: Optional[Dict[str, Any]] = None


class ConfigManager:
    """Manages configuration loading and validation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
        self._setup_logging()

    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")

        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)

        # Replace environment variable placeholders
        config = self._resolve_env_vars(config)
        return config

    def _resolve_env_vars(self, config: Any) -> Any:
        """Recursively resolve environment variables in config."""
        if isinstance(config, dict):
            return {k: self._resolve_env_vars(v) for k, v in config.items()}
        elif isinstance(config, list):
            return [self._resolve_env_vars(item) for item in config]
        elif isinstance(config, str) and config.startswith('${') and config.endswith('}'):
            env_var = config[2:-1]
            return os.getenv(env_var, "")
        return config

    def _setup_logging(self):
        """Setup logging configuration."""
        log_config = self.config.get('logging', {})
        level = getattr(logging, log_config.get('level', 'INFO'))

        handlers = []
        if log_config.get('console_output', True):
            handlers.append(logging.StreamHandler())
        if log_config.get('log_file'):
            handlers.append(logging.FileHandler(log_config['log_file']))

        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=handlers
        )

    def get(self, key_path: str, default: Any = None) -> Any:
        """Get configuration value using dot notation."""
        keys = key_path.split('.')
        value = self.config
        for key in keys:
            if isinstance(value, dict):
                value = value.get(key)
            else:
                return default
            if value is None:
                return default
        return value


class ScenarioLoader:
    """Loads and parses scenario files."""

    def __init__(self, scenarios_folder: str = "./scenarios"):
        self.scenarios_folder = scenarios_folder
        self.logger = logging.getLogger(self.__class__.__name__)

    def load_scenario(self, filename: str) -> ScenarioData:
        """Load a single scenario file."""
        filepath = os.path.join(self.scenarios_folder, filename)

        if not os.path.exists(filepath):
            raise FileNotFoundError(f"Scenario file not found: {filepath}")

        self.logger.info(f"Loading scenario: {filename}")

        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()

        return self._parse_scenario(content, filename)

    def _parse_scenario(self, content: str, filename: str) -> ScenarioData:
        """Parse scenario content."""
        lines = content.strip().split('\n')

        # Parse metadata from header
        metadata = {}
        user = "User"
        system_prompt = "You are a helpful and supportive chatbot assistant."
        messages = []

        in_header = True
        for line in lines:
            line = line.strip()

            if line == '---':
                in_header = False
                continue

            if in_header:
                if line.startswith('USER:'):
                    user = line.split('USER:', 1)[1].strip()
                elif line.startswith('SYSTEM_PROMPT:'):
                    system_prompt = line.split('SYSTEM_PROMPT:', 1)[1].strip()
                elif line.startswith('#'):
                    # Comment line, skip
                    continue
                elif ':' in line:
                    # Custom metadata
                    key, value = line.split(':', 1)
                    metadata[key.strip()] = value.strip()
            else:
                # Message content
                if line and not line.startswith('#'):
                    messages.append(line)

        scenario_id = Path(filename).stem

        return ScenarioData(
            scenario_id=scenario_id,
            user=user,
            system_prompt=system_prompt,
            messages=messages,
            metadata=metadata
        )

    def load_all_scenarios(self) -> List[ScenarioData]:
        """Load all scenario files from the folder."""
        if not os.path.exists(self.scenarios_folder):
            self.logger.warning(f"Scenarios folder not found: {self.scenarios_folder}")
            return []

        scenarios = []
        for filename in os.listdir(self.scenarios_folder):
            if filename.endswith('.txt'):
                try:
                    scenario = self.load_scenario(filename)
                    scenarios.append(scenario)
                except Exception as e:
                    self.logger.error(f"Error loading scenario {filename}: {e}")

        self.logger.info(f"Loaded {len(scenarios)} scenarios")
        return scenarios


class LLMProvider(ABC):
    """Wrapper abstract base class for LLM providers."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        self.model_name = model_name
        self.config = config
        self.logger = logging.getLogger(f"{self.__class__.__name__}.{model_name}")

    @abstractmethod
    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Generate a response from the LLM."""
        pass

    @property
    @abstractmethod
    def provider_name(self) -> str:
        """Return the provider name."""
        pass


class OpenAIProvider(LLMProvider):
    """OpenAI LLM provider using LangChain."""

    def __init__(self, model_name: str, config: Dict[str, Any], api_key: str):
        super().__init__(model_name, config)
        try:
            from langchain_openai import ChatOpenAI
        except ImportError as e:
            raise ImportError("Missing dependency for OpenAI provider. Please install: pip install langchain-openai") from e

        self.llm = ChatOpenAI(
            model=model_name,
            api_key=api_key,
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 1000),
            top_p=config.get('top_p', 1.0)
        )

    @property
    def provider_name(self) -> str:
        return "openai"

    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Generate response using OpenAI."""
        try:
            from langchain.schema import HumanMessage, SystemMessage, AIMessage
        except ImportError as e:
            raise ImportError("Missing langchain schema. Please install: pip install langchain-community") from e

        langchain_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=msg['content']))

        response = self.llm.invoke(langchain_messages)
        return response.content


class HuggingFaceProvider(LLMProvider):
    """HuggingFace local/self-hosted model provider via transformers."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        try:
            from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
        except Exception as e:
            raise ImportError("Missing transformers package. Install with: pip install transformers") from e

        import torch

        # Resolve model identifier
        self.model_id = self.config.get('model_id') or (
            'sshleifer/tiny-gpt2' if self.model_name == 'placeholder' else self.model_name
        )

        # Check if gated and warn user
        if self.config.get('gated', False):
            self.logger.warning(
                f"Model '{self.model_id}' is GATED and requires Hugging Face authentication.\n"
                f"If download fails:\n"
                f"  1. Visit https://huggingface.co/{self.model_id} and accept the terms\n"
                f"  2. Run: huggingface-cli login"
            )

        # Device and dtype configuration
        device_pref = self.config.get('device')
        self.device = device_pref if device_pref in ('cpu', 'cuda') else ('cuda' if torch.cuda.is_available() else 'cpu')
        dtype_str = self.config.get('torch_dtype', 'auto')
        torch_dtype = None if dtype_str == 'auto' else getattr(torch, dtype_str, None)

        # Build model kwargs
        trust_remote = self.config.get('trust_remote_code', True)
        model_kwargs: Dict[str, Any] = {
            'trust_remote_code': trust_remote,
            'low_cpu_mem_usage': True,
        }

        # Pass through optional per-model setting to force/disable safetensors
        if 'use_safetensors' in self.config:
            model_kwargs['use_safetensors'] = bool(self.config.get('use_safetensors'))

        # If flash_attn is not available, prefer eager attention for compatibility
        try:
            import flash_attn  # type: ignore  # noqa: F401
            _has_flash_attn = True
        except Exception:
            _has_flash_attn = False
        if not _has_flash_attn and 'attn_implementation' not in self.config:
            model_kwargs['attn_implementation'] = 'eager'
        
        # Configure quantization using BitsAndBytesConfig if requested
        if self.config.get('load_in_4bit') or self.config.get('load_in_8bit'):
            bnb_config_kwargs = {}
            
            if self.config.get('load_in_4bit'):
                bnb_config_kwargs['load_in_4bit'] = True
                bnb_config_kwargs['bnb_4bit_use_double_quant'] = self.config.get('bnb_4bit_use_double_quant', True)
                bnb_config_kwargs['bnb_4bit_quant_type'] = self.config.get('bnb_4bit_quant_type', 'nf4')
                if torch_dtype is not None:
                    bnb_config_kwargs['bnb_4bit_compute_dtype'] = torch_dtype
            
            if self.config.get('load_in_8bit'):
                bnb_config_kwargs['load_in_8bit'] = True
            
            model_kwargs['quantization_config'] = BitsAndBytesConfig(**bnb_config_kwargs)
            model_kwargs['device_map'] = 'auto'
            # Enable offloading folder to help with tight GPU memory
            try:
                offload_dir = os.path.join('.', 'offload', self.model_name)
                os.makedirs(offload_dir, exist_ok=True)
                model_kwargs['offload_folder'] = offload_dir
            except Exception:
                pass
        else:
            if torch_dtype is not None:
                model_kwargs['torch_dtype'] = torch_dtype
            if self.device == 'cuda':
                model_kwargs['device_map'] = 'auto'

        # Record whether quantization is in use (affects input device handling)
        self._uses_quant = 'quantization_config' in model_kwargs

        # Load tokenizer/model with safetensors fallback
        self.logger.info(f"Loading model: {self.model_id}")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_id, trust_remote_code=trust_remote)
        
        # Load model directly (no retry shim)
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, **model_kwargs)
        
        # Handle pad_token_id for open-ended generation
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id
            self.logger.info("Set pad_token_id to eos_token_id")

        # Generation parameters
        self.gen_kwargs = {
            'max_new_tokens': self.config.get('max_new_tokens', 256),
            'temperature': self.config.get('temperature', 0.7),
            'top_p': self.config.get('top_p', 0.95),
            'do_sample': self.config.get('do_sample', True),
            'eos_token_id': self.tokenizer.eos_token_id,
        }

        # Prompting options
        self.use_chat_template = bool(self.config.get('use_chat_template', True))
        self.persona_prompt = (self.config.get('system_prompt') or '').strip()

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        from transformers import TextStreamer
        import torch

        # Merge persona and scenario system prompts
        effective_system = self.persona_prompt
        if system_prompt:
            effective_system = f"{effective_system}\n{system_prompt}" if effective_system else system_prompt

        # Build chat-style message list
        chat_messages: List[Dict[str, str]] = []
        if effective_system:
            chat_messages.append({'role': 'system', 'content': effective_system})
        for m in messages:
            role = m.get('role')
            if role in ('user', 'assistant'):
                chat_messages.append({'role': role, 'content': m.get('content', '')})

        # Tokenize with chat template if available, else simple concat
        if self.use_chat_template and hasattr(self.tokenizer, 'apply_chat_template'):
            try:
                # Prefer dict output so we can obtain attention_mask too
                result = self.tokenizer.apply_chat_template(
                    chat_messages,
                    add_generation_prompt=True,
                    tokenize=True,
                    return_dict=True,
                    return_tensors='pt'
                )
                # Ensure we have tensors for both ids and mask
                input_ids = result["input_ids"]
                attention_mask = result.get("attention_mask")
                if attention_mask is None:
                    # Construct a full-ones attention mask if not provided
                    attention_mask = torch.ones_like(input_ids)
            except Exception:
                prompt_text = '\n'.join([f"{mm['role']}: {mm['content']}" for mm in chat_messages])
                encoded = self.tokenizer(prompt_text, return_tensors='pt')
                input_ids = encoded.input_ids
                attention_mask = encoded.attention_mask if hasattr(encoded, 'attention_mask') else torch.ones_like(input_ids)
        else:
            prompt_text = '\n'.join([f"{mm['role']}: {mm['content']}" for mm in chat_messages])
            encoded = self.tokenizer(prompt_text, return_tensors='pt')
            input_ids = encoded.input_ids
            attention_mask = encoded.attention_mask if hasattr(encoded, 'attention_mask') else torch.ones_like(input_ids)

        # When using quantization with device_map='auto', do NOT move tensors manually
        # Quantized models are already on the correct device
        if not self._uses_quant:
            device = next(self.model.parameters()).device
            input_ids = input_ids.to(device)
            attention_mask = attention_mask.to(device)

        with torch.no_grad():
            output_ids = self.model.generate(input_ids=input_ids, attention_mask=attention_mask, **self.gen_kwargs)

        gen_ids = output_ids[0][input_ids.shape[-1]:]
        text = self.tokenizer.decode(gen_ids, skip_special_tokens=True)
        return text.strip()


class NoKeyAPIProvider(LLMProvider):
    """Fallback provider when a cloud API key is missing."""

    def __init__(self, provider: str, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        self._provider = provider

    @property
    def provider_name(self) -> str:
        return self._provider

    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        return f"PLACEHOLDER, NO KEY FOR {self._provider.upper()} MODEL"


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(provider_name: str, model_name: str, config: 'ConfigManager') -> LLMProvider:
        # Avoid dot-notation for model_name keys that may contain dots (e.g., qwen2.5-7b)
        models_root = config.get('models', {}) or {}
        provider_root = models_root.get(provider_name, {}) if isinstance(models_root, dict) else {}
        model_config = provider_root.get(model_name, {}) if isinstance(provider_root, dict) else {}

        if provider_name == 'openai':
            api_key = config.get('api_keys.openai')
            if not api_key:
                logging.getLogger('LLMProviderFactory').warning('OpenAI API key missing; using placeholder provider')
                return NoKeyAPIProvider('openai', model_name, model_config)
            return OpenAIProvider(model_name, model_config, api_key)

        elif provider_name == 'anthropic':
            api_key = config.get('api_keys.anthropic')
            if not api_key:
                logging.getLogger('LLMProviderFactory').warning('Anthropic API key missing; using placeholder provider')
                return NoKeyAPIProvider('anthropic', model_name, model_config)
            return AnthropicProvider(model_name, model_config, api_key)

        elif provider_name == 'google':
            api_key = config.get('api_keys.google')
            if not api_key:
                logging.getLogger('LLMProviderFactory').warning('Google API key missing; using placeholder provider')
                return NoKeyAPIProvider('google', model_name, model_config)
            return GoogleProvider(model_name, model_config, api_key)

        elif provider_name == 'huggingface':
            return HuggingFaceProvider(model_name, model_config)

        else:
            raise ValueError(f"Unknown provider: {provider_name}")

class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) LLM provider using LangChain."""

    def __init__(self, model_name: str, config: Dict[str, Any], api_key: str):
        super().__init__(model_name, config)
        try:
            from langchain_anthropic import ChatAnthropic
        except ImportError as e:
            raise ImportError("Missing dependency for Anthropic provider. Please install: pip install langchain-anthropic") from e

        self.llm = ChatAnthropic(
            model=model_name,
            anthropic_api_key=api_key,
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 1000),
            top_p=config.get('top_p', 1.0)
        )

    @property
    def provider_name(self) -> str:
        return "anthropic"

    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Generate response using Anthropic."""
        try:
            from langchain.schema import HumanMessage, SystemMessage, AIMessage
        except ImportError as e:
            raise ImportError("Missing langchain schema. Please install: pip install langchain-community") from e

        langchain_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=msg['content']))

        response = self.llm.invoke(langchain_messages)
        return response.content


class GoogleProvider(LLMProvider):
    """Google (Gemini) LLM provider using LangChain."""

    def __init__(self, model_name: str, config: Dict[str, Any], api_key: str):
        super().__init__(model_name, config)
        try:
            from langchain_google_genai import ChatGoogleGenerativeAI
        except ImportError as e:
            raise ImportError("Missing dependency for Google provider. Please install: pip install langchain-google-genai") from e

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=api_key,
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 1000),
            top_p=config.get('top_p', 1.0)
        )

    @property
    def provider_name(self) -> str:
        return "google"

    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Generate response using Google Gemini."""
        try:
            from langchain.schema import HumanMessage, SystemMessage, AIMessage
        except ImportError as e:
            raise ImportError("Missing langchain schema. Please install: pip install langchain-community") from e

        langchain_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=msg['content']))

        response = self.llm.invoke(langchain_messages)
        return response.content


class ConversationRecorder:
    """Records simulated conversations"""

    def __init__(self, output_folder: str = "./outputs/conversations", pretty_print: bool = True):
        self.output_folder_template = output_folder
        self.pretty_print = pretty_print
        self.logger = logging.getLogger(self.__class__.__name__)

    # Backward-compatible alias for legacy code that reads recorder.output_folder
    @property
    def output_folder(self) -> str:
        return self.output_folder_template

    def save_conversation(self, record: ConversationRecord) -> str:
        """Save a conversation record to disk."""
        # Replace {model} placeholder in output folder path
        output_folder = self.output_folder_template.replace('{model}', record.model_name)
        os.makedirs(output_folder, exist_ok=True)
        
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{record.scenario_id}_{record.provider}_{record.model_name}_{timestamp_str}.json"
        filepath = os.path.join(output_folder, filename)

        # Convert to dictionary
        record_dict = asdict(record)

        # Save to file
        with open(filepath, 'w', encoding='utf-8') as f:
            if self.pretty_print:
                json.dump(record_dict, f, indent=2, ensure_ascii=False)
            else:
                json.dump(record_dict, f, ensure_ascii=False)

        self.logger.info(f"Saved conversation to: {filepath}")
        return filepath

    def load_conversation(self, filepath: str) -> ConversationRecord:
        """Load a conversation record from disk."""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return ConversationRecord(**data)

class ConversationSimulator:
    """Main orchestrator for conversation simulation."""

    def __init__(self, config_path: str = "config.yaml"):
        self.config = ConfigManager(config_path)
        self.logger = logging.getLogger(self.__class__.__name__)

        # Initialize components
        scenarios_folder = self.config.get('scenarios.folder', './scenarios')
        self.scenario_loader = ScenarioLoader(scenarios_folder)

        conversations_folder = self.config.get('output.conversations_folder', './outputs/conversations')
        pretty_print = self.config.get('output.pretty_print', True)
        self.recorder = ConversationRecorder(conversations_folder, pretty_print)

        # Initialize risk analyzer if enabled
        self.risk_analyzer: Optional[Any] = None
        if self.config.get('risk_analysis.auto_analyze', False):
            try:
                from RiskAnalyzer import RiskAnalyzer
                self.risk_analyzer = RiskAnalyzer(self.config)
            except Exception as e:
                self.logger.error(f"Failed to initialize RiskAnalyzer: {e}")
                self.risk_analyzer = None

        self.logger.info("ConversationSimulator initialized")

    def simulate_batch(
        self,
        scenarios: List[ScenarioData],
        provider_model_pairs: List[tuple]
    ) -> List[ConversationRecord]:
        """Simulate multiple conversations with provider caching."""
        results = []

        total = len(scenarios) * len(provider_model_pairs)
        self.logger.info(f"Starting batch simulation: {total} conversations")

        # Cache providers to avoid reloading models for each scenario
        provider_cache: Dict[tuple, Any] = {}

        # Pre-load all providers
        self.logger.info(f"Pre-loading {len(provider_model_pairs)} model(s)...")
        for provider_name, model_name in provider_model_pairs:
            cache_key = (provider_name, model_name)
            if cache_key not in provider_cache:
                try:
                    provider = LLMProviderFactory.create_provider(provider_name, model_name, self.config)
                    provider_cache[cache_key] = provider
                    self.logger.info(f"Loaded {provider_name}/{model_name}")
                except Exception as e:
                    self.logger.error(f"Failed to load {provider_name}/{model_name}: {e}")
                    provider_cache[cache_key] = None

        # Run scenarios with cached providers
        for scenario in scenarios:
            for provider_name, model_name in provider_model_pairs:
                cache_key = (provider_name, model_name)
                provider = provider_cache.get(cache_key)
                
                if provider is None:
                    self.logger.warning(f"Skipping {scenario.scenario_id} with {provider_name}/{model_name} (failed to load)")
                    continue

                try:
                    record = self._simulate_with_provider(scenario, provider_name, model_name, provider)
                    results.append(record)
                except Exception as e:
                    self.logger.error(f"Failed to simulate {scenario.scenario_id} with {provider_name}/{model_name}: {e}")

        self.logger.info(f"Batch simulation complete: {len(results)}/{total} successful")
        return results

    def _simulate_with_provider(
        self,
        scenario: ScenarioData,
        provider_name: str,
        model_name: str,
        provider: LLMProvider
    ) -> ConversationRecord:
        """Simulate a single conversation with a pre-loaded provider."""
        self.logger.info(f"Simulating: {scenario.scenario_id} with {provider_name}/{model_name}")

        # Build conversation
        conversation = []
        for user_message in scenario.messages:
            # Add user message
            conversation.append({
                'role': 'user',
                'content': user_message
            })

            # Generate assistant response
            try:
                response = provider.generate_response(conversation, scenario.system_prompt)
                conversation.append({
                    'role': 'assistant',
                    'content': response
                })

                # Rate limiting
                time.sleep(0.1)  # Basic rate limiting

            except Exception as e:
                self.logger.error(f"Error generating response: {e}")
                conversation.append({
                    'role': 'assistant',
                    'content': f"[ERROR: {str(e)}]"
                })

        # Create record
        # Avoid dot-notation for model_name keys that may contain dots
        models_root = self.config.get('models', {}) or {}
        provider_root = models_root.get(provider_name, {}) if isinstance(models_root, dict) else {}
        model_config = provider_root.get(model_name, {}) if isinstance(provider_root, dict) else {}
        record = ConversationRecord(
            scenario_id=scenario.scenario_id,
            model_name=model_name,
            provider=provider_name,
            timestamp=datetime.now().isoformat(),
            user=scenario.user,
            system_prompt=scenario.system_prompt,
            conversation=conversation,
            model_parameters=model_config
        )

        # Save conversation
        self.recorder.save_conversation(record)

        # Analyze risk if enabled
        if self.config.get('risk_analysis.auto_analyze', False) and self.risk_analyzer:
            try:
                risk_metrics = self.risk_analyzer.analyze_conversation(record)
                record.risk_metrics = risk_metrics
                self.risk_analyzer.save_risk_report(record, risk_metrics)
            except Exception as e:
                self.logger.error(f"Risk analysis failed: {e}")

        return record

    def run_all_scenarios(self, provider_model_pairs: List[tuple]) -> List[ConversationRecord]:
        """Run all scenarios with specified models."""
        scenarios = self.scenario_loader.load_all_scenarios()

        if not scenarios:
            self.logger.warning("No scenarios found")
            return []

        return self.simulate_batch(scenarios, provider_model_pairs)


def main():
    """Main CLI entry point."""
    print("🤖 LLM Conversation Simulator\n")

    # Initialize simulator
    try:
        simulator = ConversationSimulator()
    except Exception as e:
        print(f"Error initializing simulator: {e}")
        return

    # Load scenarios
    scenarios = simulator.scenario_loader.load_all_scenarios()
    if not scenarios:
        print("No scenarios found in ./scenarios folder")
        print("Create .txt files in ./scenarios folder to get started")
        return

    # Read provider/model pairs from config, default to offline huggingface placeholder
    provider_model_pairs = simulator.config.get('run.provider_models', [('huggingface', 'placeholder')])
    print(f"Running {len(scenarios)} scenario(s) with models: {provider_model_pairs}")
    results = simulator.simulate_batch(scenarios, provider_model_pairs)
    print(f"Done. Saved {len(results)} conversation(s) to {simulator.recorder.output_folder}")


if __name__ == "__main__":
    main()
