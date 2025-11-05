import os
import json
from datetime import datetime
from typing import Any, Dict, List


class RiskAnalyzer:
    def __init__(self, config):
        # Accept ConfigManager or dict
        self._config = config
        self.output_folder = self._get_cfg('output.risk_reports_folder', 'outputs/risk_reports')
        os.makedirs(self.output_folder, exist_ok=True)

        # Analysis settings
        self.embedding_model = self._get_cfg('analysis.embedding_model', 'all-MiniLM-L6-v2')
        self.enable_t5 = self._get_cfg('analysis.enable_t5_qualitative', False)
        self.t5_name_or_path = self._get_cfg('analysis.t5_model_name_or_path', 'google/flan-t5-base')

    def _get_cfg(self, key: str, default: Any = None) -> Any:
        if hasattr(self._config, 'get') and callable(self._config.get):
            return self._config.get(key, default)
        # Fallback if plain dict
        parts = key.split('.')
        cur = self._config
        for p in parts:
            if isinstance(cur, dict) and p in cur:
                cur = cur[p]
            else:
                return default
        return cur

    def analyze_conversation(self, record: Any) -> List[Dict[str, Any]]:
        """
        Given a ConversationRecord-like object (dict-like or dataclass), run transcript analysis
        on assistant messages and return a list of per-paragraph metric dicts.
        """
        # Extract assistant messages as paragraphs
        conversation = getattr(record, 'conversation', None) or record.get('conversation', [])
        scenario = getattr(record, 'scenario_id', None) or record.get('scenario_id', 'unknown')
        user = getattr(record, 'user', None) or record.get('user', 'User')

        assistant_paragraphs = [t['content'] for t in conversation if t.get('role') == 'assistant']

        # Lazy import heavy analyzer only when needed
        from LLMTranscriptAnalyzer import LLMTranscriptAnalyzer  # type: ignore

        analyzer = LLMTranscriptAnalyzer(
            transcript=assistant_paragraphs,
            scenario=scenario,
            user=user,
            embedding_model_name=self.embedding_model,
            enable_t5_qualitative=self.enable_t5,
            t5_model_name_or_path=self.t5_name_or_path,
        )
        # Configure analyzer by monkey-patching module-level preferences if exposed;
        # current analyzer reads global models at import time. If it is refactored later to accept
        # config directly, wire settings here.
        # For now, just call analyze().
        metrics = analyzer.analyze()
        return metrics

    def save_risk_report(self, record: Any, metrics: List[Dict[str, Any]]) -> str:
        ts = datetime.now().strftime('%Y%m%d_%H%M%S')
        scenario = getattr(record, 'scenario_id', None) or record.get('scenario_id', 'unknown')
        provider = getattr(record, 'provider', None) or record.get('provider', 'unknown')
        model = getattr(record, 'model_name', None) or record.get('model_name', 'unknown')
        filename = f"{scenario}_{provider}_{model}_{ts}.json"
        path = os.path.join(self.output_folder, filename)

        payload = {
            'scenario_id': scenario,
            'provider': provider,
            'model_name': model,
            'timestamp': getattr(record, 'timestamp', None) or record.get('timestamp', None),
            'metrics': metrics,
        }
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        return path
