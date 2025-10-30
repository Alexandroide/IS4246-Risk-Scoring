"""
LLM Conversation Simulator
Simulates conversations with different LLM models using scenarios from text files
and records responses for risk analysis.
"""

import os
import json
import yaml
import logging
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional, Any
from abc import ABC, abstractmethod
from dataclasses import dataclass, asdict
import time
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.llms import HuggingFaceHub
from langchain_cohere import ChatCohere
from langchain.schema import HumanMessage, SystemMessage, AIMessage

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
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed")

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
        langchain_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=msg['content']))

        response = self.llm.invoke(langchain_messages)
        return response.content


class AnthropicProvider(LLMProvider):
    """Anthropic (Claude) LLM provider using LangChain."""

    def __init__(self, model_name: str, config: Dict[str, Any], api_key: str):
        super().__init__(model_name, config)
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed")

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
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed")

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
        langchain_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=msg['content']))

        response = self.llm.invoke(langchain_messages)
        return response.content


class CohereProvider(LLMProvider):
    """Cohere LLM provider using LangChain."""

    def __init__(self, model_name: str, config: Dict[str, Any], api_key: str):
        super().__init__(model_name, config)
        if not LANGCHAIN_AVAILABLE:
            raise ImportError("LangChain not installed")

        self.llm = ChatCohere(
            model=model_name,
            cohere_api_key=api_key,
            temperature=config.get('temperature', 0.7),
            max_tokens=config.get('max_tokens', 1000)
        )

    @property
    def provider_name(self) -> str:
        return "cohere"

    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Generate response using Cohere."""
        langchain_messages = [SystemMessage(content=system_prompt)]

        for msg in messages:
            if msg['role'] == 'user':
                langchain_messages.append(HumanMessage(content=msg['content']))
            elif msg['role'] == 'assistant':
                langchain_messages.append(AIMessage(content=msg['content']))

        response = self.llm.invoke(langchain_messages)
        return response.content


class HuggingFaceProvider(LLMProvider):
    """HuggingFace local model provider."""

    def __init__(self, model_name: str, config: Dict[str, Any]):
        super().__init__(model_name, config)
        # For local models, implement using transformers directly
        # This is a placeholder - adapt based on your local model setup
        self.logger.warning("HuggingFace provider is a placeholder. Implement based on your model setup.")

    @property
    def provider_name(self) -> str:
        return "huggingface"

    def generate_response(self, messages: List[Dict[str, str]], system_prompt: str) -> str:
        """Generate response using local HuggingFace model."""
        # Placeholder implementation
        # You would load your local model here
        self.logger.warning("Using placeholder response for HuggingFace")
        return "This is a placeholder response. Implement HuggingFace provider for your local model."


class LLMProviderFactory:
    """Factory for creating LLM providers."""

    @staticmethod
    def create_provider(provider_name: str, model_name: str, config: ConfigManager) -> LLMProvider:
        """Create an LLM provider instance."""
        model_config = config.get(f'models.{provider_name}.{model_name}', {})

        if provider_name == 'openai':
            api_key = config.get('api_keys.openai')
            if not api_key:
                raise ValueError("OpenAI API key not found in configuration")
            return OpenAIProvider(model_name, model_config, api_key)

        elif provider_name == 'anthropic':
            api_key = config.get('api_keys.anthropic')
            if not api_key:
                raise ValueError("Anthropic API key not found in configuration")
            return AnthropicProvider(model_name, model_config, api_key)

        elif provider_name == 'google':
            api_key = config.get('api_keys.google')
            if not api_key:
                raise ValueError("Google API key not found in configuration")
            return GoogleProvider(model_name, model_config, api_key)

        elif provider_name == 'cohere':
            api_key = config.get('api_keys.cohere')
            if not api_key:
                raise ValueError("Cohere API key not found in configuration")
            return CohereProvider(model_name, model_config, api_key)

        elif provider_name == 'huggingface':
            return HuggingFaceProvider(model_name, model_config)

        else:
            raise ValueError(f"Unknown provider: {provider_name}")


class ConversationRecorder:
    """Records simulated conversations"""

    def __init__(self, output_folder: str = "./outputs/conversations", pretty_print: bool = True):
        self.output_folder = output_folder
        self.pretty_print = pretty_print
        self.logger = logging.getLogger(self.__class__.__name__)

        # Create output folder if it doesn't exist
        os.makedirs(self.output_folder, exist_ok=True)

    def save_conversation(self, record: ConversationRecord) -> str:
        """Save a conversation record to disk."""
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{record.scenario_id}_{record.provider}_{record.model_name}_{timestamp_str}.json"
        filepath = os.path.join(self.output_folder, filename)

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

        self.logger.info("ConversationSimulator initialized")

    def simulate_conversation(
        self,
        scenario: ScenarioData,
        provider_name: str,
        model_name: str
    ) -> ConversationRecord:
        """Simulate a single conversation."""
        self.logger.info(f"Simulating: {scenario.scenario_id} with {provider_name}/{model_name}")

        # Create provider
        provider = LLMProviderFactory.create_provider(provider_name, model_name, self.config)

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
        model_config = self.config.get(f'models.{provider_name}.{model_name}', {})
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
        if self.config.get('risk_analysis.auto_analyze', True):
            risk_metrics = self.risk_analyzer.analyze_conversation(record)
            record.risk_metrics = risk_metrics
            self.risk_analyzer.save_risk_report(record, risk_metrics)

        return record

    def simulate_batch(
        self,
        scenarios: List[ScenarioData],
        provider_model_pairs: List[tuple]
    ) -> List[ConversationRecord]:
        """Simulate multiple conversations."""
        results = []

        total = len(scenarios) * len(provider_model_pairs)
        self.logger.info(f"Starting batch simulation: {total} conversations")

        for scenario in scenarios:
            for provider_name, model_name in provider_model_pairs:
                try:
                    record = self.simulate_conversation(scenario, provider_name, model_name)
                    results.append(record)
                except Exception as e:
                    self.logger.error(f"Failed to simulate {scenario.scenario_id} with {provider_name}/{model_name}: {e}")

        self.logger.info(f"Batch simulation complete: {len(results)}/{total} successful")
        return results

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

    # Example usage
    print("Available usage patterns:\n")
    print("1. Simulate single scenario:")
    print("   scenario = simulator.scenario_loader.load_scenario('example.txt')")
    print("   record = simulator.simulate_conversation(scenario, 'openai', 'gpt-4')\n")

    print("2. Simulate all scenarios with multiple models:")
    print("   models = [('openai', 'gpt-4'), ('anthropic', 'claude-3-sonnet')]")
    print("   results = simulator.run_all_scenarios(models)\n")

    print("3. Load and analyze existing conversation:")
    print("   record = simulator.recorder.load_conversation('path/to/file.json')")
    print("   metrics = simulator.risk_analyzer.analyze_conversation(record)\n")

    # Check for scenarios
    scenarios = simulator.scenario_loader.load_all_scenarios()
    if scenarios:
        print(f"  Found {len(scenarios)} scenario(s):")
        for scenario in scenarios:
            print(f"   - {scenario.scenario_id} ({len(scenario.messages)} messages)")
    else:
        print("   No scenarios found in ./scenarios folder")
        print("   Create .txt files in ./scenarios folder to get started")


if __name__ == "__main__":
    main()
