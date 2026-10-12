"""Local configuration example for the MemWire release bundle.

Copy this file to ``config.py`` and replace only local model paths and the
OpenAI-compatible endpoint. Do not put provider credentials in this file when
using a local vLLM server.
"""

# ============================================================================
# LLM Configuration
# ============================================================================

# A local vLLM server can use any non-empty placeholder key.
OPENAI_API_KEY = "memwire-local"

# Custom OpenAI Base URL (optional)
# Set to None to use default OpenAI endpoint
# Examples:
#   - Qwen/Alibaba: "https://dashscope.aliyuncs.com/compatible-mode/v1"
#   - Azure OpenAI: "https://YOUR-RESOURCE.openai.azure.com/openai/deployments/YOUR-DEPLOYMENT"
#   - Local server: "http://localhost:8000/v1"
#   - OpenAI (default): None
OPENAI_BASE_URL = "http://127.0.0.1:8000/v1"

# LLM Model name
# Examples: "gpt-4.1-mini", "gpt-4.1", "qwen3-max", "qwen-plus-2025-07-28"
LLM_MODEL = "qwen3.5-9b"
TOKENIZER_MODEL_PATH = "/path/to/Qwen3.5-9B"

# Embedding model (local, no API needed)
EMBEDDING_MODEL = "/path/to/Qwen3-Embedding-0.6B"
EMBEDDING_DIMENSION = 1024  # For Qwen3: up to 1024, supports 32-1024
EMBEDDING_CONTEXT_LENGTH = 32768  # Qwen3 supports 32k context


# ============================================================================
# Advanced LLM Features
# ============================================================================

# Enable deep thinking mode (for Qwen and compatible models)
# Adds extra_body={"enable_thinking": True} to API calls
# Set to False for OpenAI models (they don't support this)
ENABLE_THINKING = False

# Enable streaming responses (outputs content as it's generated)
USE_STREAMING = True

# Enable JSON format mode (ensures LLM outputs valid JSON)
# Adds response_format={"type": "json_object"} to API calls
# Helps prevent parsing failures from extra text like ```json
USE_JSON_FORMAT = True


# ============================================================================
# Memory Building Parameters
# ============================================================================

# Fixed-size cell construction over the complete temporally ordered history.
WINDOW_SIZE = 20
THREAD_TURN_LIMIT = 10
# Generation-free fixed-size cells are the paper protocol. Set True only for
# the optional construction-prompt ablation.
ENABLE_THREAD_SEGMENTATION = False

# Window overlap size (for context continuity)
OVERLAP_SIZE = 5


# ============================================================================
# MemWire routing and local activation
# ============================================================================

# ============================================================================
# Database Configuration
# ============================================================================

# Persistent cell store used by the active method
THREAD_LANCEDB_PATH = "./thread_lancedb_data"
THREAD_TABLE_NAME = "event_threads_v3"
THREAD_ROUTE_TOP_K = 16
THREAD_ROUTE_PROBE_TOP_K = 12
THREAD_ROUTE_PROBE_MARGIN = 0.02
# The paper uses r_next = (r_current + edge_weight) / 2.
THREAD_EXPANSION_SCORE_DECAY = 0.5
THREAD_EXPANSION_EDGE_WEIGHT = 0.5
THREAD_MAX_HOPS = 3
THREAD_MAX_EXPANDED = 32
THREAD_ADAPTIVE_MAX_EXPANDED = 24
# Canonical graph and finite-hop expansion names. The historical THREAD_*
# names above remain accepted as compatibility aliases.
ADDRESS_ROUTE_WIDTH = 16
ADDRESS_PROBE_WIDTH = 24
RECEPTIVE_FIELD_HOPS = 3
RECEPTIVE_FIELD_SCOPE = 32
BROAD_RECEPTIVE_FIELD_SCOPE = 24
THREAD_GRAPH_NEIGHBORS = 5
THREAD_GRAPH_MIN_SIM = 0.25
THREAD_GRAPH_MODE = "operator-aware"
THREAD_OPERATOR_CANDIDATE_MULTIPLIER = 3
THREAD_OPERATOR_SKETCH_DIM = 6
THREAD_OPERATOR_OBJECTIVE_SCALE = 1.0
THREAD_OPERATOR_HORIZON = 3
THREAD_OPERATOR_LAMBDA_BAL = 1.0 / 40.0
THREAD_OPERATOR_EPS = 1e-6
THREAD_OPERATOR_SWAP_SEARCH_WIDTH = 256
THREAD_OPERATOR_MAX_SWAPS = 12



# ============================================================================
# Parallel Processing Configuration
# ============================================================================

# Memory Building Parallel Processing
ENABLE_PARALLEL_PROCESSING = True
MAX_PARALLEL_WORKERS = 16  # Number of parallel workers for memory building

# Retrieval Parallel Processing  
ENABLE_PARALLEL_RETRIEVAL = True
MAX_RETRIEVAL_WORKERS = 8  # Number of parallel workers for retrieval queries

# Optional one-address LLM planner; disabled in the paper protocol.
ENABLE_THREAD_ADDRESS_PLANNER = False
ENABLE_MEMORY_WORKSPACE = True
# The transient field determines visibility; this is only an external prompt
# serialization guard and never participates in the state dynamics.
WORKSPACE_PROMPT_GUARD_TOKENS = 1900
# At most K original source turns are selected after cell expansion.
WORKSPACE_MAX_READOUT_CELLS = 4
SOURCE_READOUT_K = 4
WORKSPACE_READOUT_TOKEN_BUDGET = 1600
WORKSPACE_READOUT_MODE = "query_similarity"
WORKSPACE_MIN_VISIBLE_CELLS = 1
# Compact answer-side rendering removes repeated metadata, not source cells.
WORKSPACE_COMPACT_SERIALIZATION = True
WORKSPACE_SOURCE_ONLY_SERIALIZATION = True
WORKSPACE_SOURCE_CHAR_BUDGET = 900
ANSWER_PROMPT_STYLE = "minimal"
MAX_TOKENS = 128
ANSWER_MAX_INPUT_TOKENS = 1800
WORKSPACE_MAX_RELATIONS = 12
# Shared local recurrent update and state readout.
WORKSPACE_ROLLOUT_STEPS = 20
WORKSPACE_STEP_SIZE = 0.35
WORKSPACE_CONVERGENCE_TOL = 1e-3
WORKSPACE_VISIBILITY_FLOOR = 0.30
WORKSPACE_COMPATIBILITY_SCALE = 1.35
WORKSPACE_INHIBITION_SCALE = 0.35
WORKSPACE_NODE_SEED_WEIGHT = 0.80


# ============================================================================
# LLM-as-Judge Configuration (not used yet)
# ============================================================================

# Judge is intentionally disabled in the release default. Enable it only in a
# separate evaluation configuration, never in the primary method run.
JUDGE_API_KEY = ""

# Judge LLM Base URL (optional - if None, uses OPENAI_BASE_URL)
# Example: Use cheaper endpoint for evaluation
JUDGE_BASE_URL = None

# Judge LLM Model (optional - if None, uses LLM_MODEL)
JUDGE_MODEL = ""

# Judge specific settings
JUDGE_ENABLE_THINKING = False  # Usually false for evaluation tasks
JUDGE_USE_STREAMING = False    # Usually false for evaluation
JUDGE_TEMPERATURE = 0.3        

# Example configurations:
# 1. Use cheaper model for judge evaluation:
#    JUDGE_MODEL = "gpt-4.1-mini"
#
# 2. Use different API provider for judge:
#    JUDGE_API_KEY = "your-judge-api-key"
#    JUDGE_BASE_URL = "https://api.different-provider.com/v1"
#    JUDGE_MODEL = "different-provider-model"
#
# 3. Use Qwen for judge (if available):
#    JUDGE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"
#    JUDGE_MODEL = "qwen-plus-2025-09-11"
