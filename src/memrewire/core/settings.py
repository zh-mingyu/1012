"""
MemWire runtime settings.

Resolution order for each attribute:
  1. User's top-level ``config.py`` (if importable on sys.path)
  2. Environment variable of the same name
  3. Built-in default
"""
import os


_DEFAULTS = {
    "OPENAI_API_KEY": "",
    "OPENAI_BASE_URL": None,
    "OPENROUTER_API_KEY": "",
    "OPENROUTER_BASE_URL": None,
    "LLM_MODEL": "gpt-4.1-mini",
    "TOKENIZER_MODEL_PATH": None,
    "EMBEDDING_MODEL": "Qwen/Qwen3-Embedding-0.6B",
    "EMBEDDING_DIMENSION": 1024,
    "EMBEDDING_CONTEXT_LENGTH": 32768,
    "ENABLE_THINKING": False,
    "USE_STREAMING": True,
    "USE_JSON_FORMAT": True,
    # Optional answer post-processing is disabled in the paper protocol. The
    # canonical prompt carries the temporal instruction directly to the model.
    "ENABLE_TEMPORAL_CALIBRATION": False,
    "ENABLE_COUNT_ANSWER_NORMALIZATION": False,
    "TEMPERATURE": 0.3,
    "MAX_TOKENS": 4096,
    "WINDOW_SIZE": 20,
    "THREAD_TURN_LIMIT": 10,
    # The paper's canonical construction is generation-free: split the
    # complete ordered history into fixed-size cells and preserve every turn.
    "ENABLE_THREAD_SEGMENTATION": False,
    "THREAD_LANCEDB_PATH": "./thread_lancedb_data",
    "THREAD_TABLE_NAME": "event_threads_v3",
    "THREAD_ROUTE_TOP_K": 16,
    "THREAD_ROUTE_PROBE_TOP_K": 12,
    "THREAD_ROUTE_PROBE_MARGIN": 0.02,
    # Query-time routing uses the paper's averaging recurrence
    # r_{t+1}=(r_t+w_t)/2.
    "THREAD_EXPANSION_SCORE_DECAY": 0.5,
    "THREAD_EXPANSION_EDGE_WEIGHT": 0.5,
    # Compatibility aliases for older launchers. The canonical retriever
    # scores every cell and applies the paper's mean-plus-standard-deviation
    # seed rule; it does not use an adaptive probe tail.
    "ADDRESS_ROUTE_WIDTH": 16,
    "ADDRESS_PROBE_WIDTH": 24,
    "RECEPTIVE_FIELD_HOPS": 3,
    "RECEPTIVE_FIELD_SCOPE": 32,
    "BROAD_RECEPTIVE_FIELD_SCOPE": 24,
    "THREAD_MAX_HOPS": 3,
    "THREAD_MAX_EXPANDED": 32,
    "THREAD_ADAPTIVE_MAX_EXPANDED": 24,
    "THREAD_GRAPH_NEIGHBORS": 5,
    "THREAD_GRAPH_MIN_SIM": 0.25,
    # MemRewire construction controls. The default uses the initial semantic
    # edge budget and degree sequence, then applies finite two-switch search.
    "THREAD_GRAPH_MODE": "operator-aware",
    "THREAD_OPERATOR_CANDIDATE_MULTIPLIER": 3,
    "THREAD_OPERATOR_SKETCH_DIM": 6,
    "THREAD_OPERATOR_OBJECTIVE_SCALE": 1.0,
    "THREAD_OPERATOR_HORIZON": 3,
    "THREAD_OPERATOR_LAMBDA_BAL": 0.025,
    "THREAD_OPERATOR_EPS": 1e-6,
    # Bounded exact evaluations for deterministic degree-preserving 2-switch
    # rewiring at write time. They never affect query-time computation.
    "THREAD_OPERATOR_SWAP_SEARCH_WIDTH": 256,
    "THREAD_OPERATOR_MAX_SWAPS": 12,
    "ENABLE_THREAD_ADDRESS_PLANNER": False,
    "ENABLE_PARALLEL_PROCESSING": True,
    "MAX_PARALLEL_WORKERS": 16,
    "ENABLE_PARALLEL_RETRIEVAL": True,
    "MAX_RETRIEVAL_WORKERS": 8,
    "ENABLE_MEMORY_WORKSPACE": True,
    # Canonical source-turn readout budget K.
    "WORKSPACE_PROMPT_GUARD_TOKENS": 1900,
    # Capacity of the final field readout. The recurrent state itself remains
    # dense; this bound controls only what is exposed to answer synthesis.
    # K is a source-turn budget, not a cell budget.
    "WORKSPACE_MAX_READOUT_CELLS": 4,
    "SOURCE_READOUT_K": 4,
    "WORKSPACE_READOUT_TOKEN_BUDGET": 1600,
    "WORKSPACE_MIN_VISIBLE_CELLS": 1,
    "WORKSPACE_READOUT_MODE": "query_similarity",
    # Keep the answer interface compact without changing the selected field.
    # This only removes repeated provenance scaffolding from the prompt;
    # source text, roles, dates, and relations remain available.
    "WORKSPACE_COMPACT_SERIALIZATION": True,
    # Optional low-token ablation: expose source text only and use the
    # minimal answer contract. It is disabled for the reference protocol.
    "WORKSPACE_SOURCE_ONLY_SERIALIZATION": True,
    "WORKSPACE_SOURCE_CHAR_BUDGET": 900,
    "ANSWER_PROMPT_STYLE": "minimal",
    "WORKSPACE_MAX_RELATIONS": 12,
    # Transient assembly dynamics.
    "WORKSPACE_ROLLOUT_STEPS": 10,
    "WORKSPACE_STEP_SIZE": 0.35,
    "WORKSPACE_CONVERGENCE_TOL": 1e-3,
    "WORKSPACE_VISIBILITY_FLOOR": 0.30,
    "WORKSPACE_COMPATIBILITY_SCALE": 1.35,
    "WORKSPACE_INHIBITION_SCALE": 0.35,
    "WORKSPACE_NODE_SEED_WEIGHT": 0.80,
    "ANSWER_MAX_INPUT_TOKENS": None,
}


def _coerce(default, raw):
    if default is None or isinstance(default, str):
        return raw
    if isinstance(default, bool):
        return raw.strip().lower() in ("1", "true", "yes", "on")
    if isinstance(default, int):
        return int(raw)
    if isinstance(default, float):
        return float(raw)
    return raw


class Settings:
    def __init__(self):
        try:
            import config as _user_config
        except ImportError:
            _user_config = None
        self._user_config = _user_config

    def __getattr__(self, name):
        # __getattr__ only fires when normal lookup fails, so no recursion via _user_config.
        user_cfg = self.__dict__.get("_user_config")
        if user_cfg is not None and hasattr(user_cfg, name):
            return getattr(user_cfg, name)
        env_val = os.getenv(name)
        if name in _DEFAULTS:
            if env_val is not None:
                return _coerce(_DEFAULTS[name], env_val)
            return _DEFAULTS[name]
        if env_val is not None:
            return env_val
        raise AttributeError(
            f"Setting {name!r} is not defined. Set it in config.py or as env var."
        )


settings = Settings()
