# IS4246 Risk Scoring – Simulator & Analyzer

- The goal of the risk-scoring is to prove how dangerous the LLMs can be, based on the potential responses that it provides to users who go to LLMs as a source of help in times of distress


## Quickstart (offline-first)

- **1) Install dependencies**
  - `uv venv`
  - `source ./.venv/bin/activate` (or `.venv\Scripts\activate` on Windows)
  - `uv sync`

- **2) (Optional) Authenticate with Hugging Face**
  - The default models are **ungated** and work without authentication.
  - If you add gated models later, run `hf auth login` and accept terms on the model pages.

### Using `uv` with project metadata

- `uv` automatically detects `pyproject.toml` and `uv.lock`.
- To sync environment with `uv`: `uv sync`

- **3) Configure**
  - A default `config.yaml` is provided and runs offline by default.
  - Key defaults:
    - `run.provider_models` runs four ungated self-hosted models:
      - `deepseek-r1-distill-qwen-1.5b` (1.5B params)
      - `gpt-oss-20b` (20B params, 4-bit quantized)
      - `llama-3.2-1b` (1B params)
      - `qwen3-0.6b` (0.6B params, ultra-lightweight)
    - Each model writes conversations to `outputs/{model}/conversations`
    - `risk_analysis.auto_analyze: false`
    - Scenarios folder: `scenarios/`

- **4) Run the simulator (offline)**
  - `python LLMConversationSimulator.py`
  - Each scenario is simulated against every configured model.
  - Conversations will be saved to per-model folders, e.g.:
    - `outputs/deepseek-r1-distill-qwen-1.5b/conversations/`
    - `outputs/gpt-oss-20b/conversations/`
    - `outputs/llama-3.2-1b/conversations/`
    - `outputs/qwen3-0.6b/conversations/`

## Optional: Enable online providers

- Supported providers: **openai (ChatGPT)**, **anthropic (Claude)**, **google (Gemini)**, **huggingface (self-hosted/local)**.
- Steps:
  - Set API keys in `config.yaml` under `api_keys`.
  - Add desired pairs to `run.provider_models`, e.g.:
    - `run.provider_models: [["google","gemini-1.5-flash"], ["openai","gpt-4o-mini"]]`
  - Install provider-specific integrations as needed:
    - `langchain-openai`, `langchain-anthropic`
    - Google/Gemini support requires manual install: `pip install langchain-google-genai`

## Optional: Risk analysis

- Turn on in `config.yaml`:
  - `risk_analysis.auto_analyze: true`
- Defaults use embedding-only metrics. To enable a small T5 model for qualitative checks:
  - Set `analysis.enable_t5_qualitative: true`
  - Optionally change `analysis.t5_model_name_or_path`.
- Risk reports will be written to `outputs/risk_reports/`.

## Project structure

- `LLMConversationSimulator.py`: orchestrates runs, providers, and saving conversations.
- `RiskAnalyzer.py`: wraps transcript analysis and persists risk reports.
- `LLMTranscriptAnalyzer.py`: computes paragraph-level metrics (embedding-based; optional T5).
- `scenarios/`: input scenario texts.
- `warm_sentences.json`, `neutral_sentences.json`: reference sentences for warmth density.
- `variables.json`: taxonomy of risk indicators (for future aggregation/reporting).

## Notes

- Default models are **ungated** and work without Hugging Face authentication.
- Models are loaded with 4-bit quantization where beneficial (e.g., `gpt-oss-20b`).
- Online providers (OpenAI, Anthropic, Google) import their dependencies lazily. If a package is missing, the simulator will show an install hint.
- If you add gated models (e.g., Llama-3.1-8B), you must accept terms on huggingface.co and run `hf auth login`.
