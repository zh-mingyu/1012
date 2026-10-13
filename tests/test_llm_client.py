import importlib.util
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from types import ModuleType, SimpleNamespace


_RELEASE_ROOT = Path(__file__).resolve().parents[1]
_ROOT = _RELEASE_ROOT / "src"
_SAVED_MODULES = {
    name: sys.modules.get(name)
    for name in (
        "memrewire",
        "memrewire.core",
        "memrewire.core.settings",
        "memrewire.core.utils",
        "memrewire.core.utils.llm_client",
        "memrewire.core.models",
        "memrewire.core.models.memory_entry",
        "memrewire.core.query_context",
        "memrewire.core.answer_generator",
    )
}


def _load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


for package, path in (
    ("memrewire", _ROOT / "memrewire"),
    ("memrewire.core", _ROOT / "memrewire" / "core"),
    ("memrewire.core.utils", _ROOT / "memrewire" / "core" / "utils"),
):
    module = ModuleType(package)
    module.__path__ = [str(path)]
    sys.modules.setdefault(package, module)

_load_module(
    "memrewire.core.settings",
    _ROOT / "memrewire" / "core" / "settings.py",
)
LLMClient = _load_module(
    "memrewire.core.utils.llm_client",
    _ROOT / "memrewire" / "core" / "utils" / "llm_client.py",
).LLMClient

models_package = ModuleType("memrewire.core.models")
models_package.__path__ = [str(_ROOT / "memrewire" / "core" / "models")]
sys.modules.setdefault("memrewire.core.models", models_package)
memory_entry_module = ModuleType("memrewire.core.models.memory_entry")
memory_entry_module.MemoryEntry = type("MemoryEntry", (), {})
sys.modules.setdefault("memrewire.core.models.memory_entry", memory_entry_module)
query_context_module = ModuleType("memrewire.core.query_context")
query_context_module.QueryContext = type("QueryContext", (), {})
sys.modules.setdefault("memrewire.core.query_context", query_context_module)
AnswerGenerator = _load_module(
    "memrewire.core.answer_generator",
    _ROOT / "memrewire" / "core" / "answer_generator.py",
).AnswerGenerator
merge_token_summaries = _load_module(
    "merge_memrewire_shards",
    _RELEASE_ROOT / "merge_memrewire_shards.py",
).merge_token_summaries

# This module uses light-weight import stubs for isolated client tests. Do not
# leave those stubs in sys.modules, otherwise collection order changes the
# imports seen by the real query_context and topology tests.
for _name, _module in _SAVED_MODULES.items():
    if _module is None:
        sys.modules.pop(_name, None)
    else:
        sys.modules[_name] = _module

# Keep the dependency-light test module usable when collected on its own. The
# isolated imports above may have started with no real settings module loaded.
if "memrewire.core.settings" not in sys.modules:
    import importlib
    importlib.import_module("memrewire.core.settings")

# The isolated answer-generator import may have captured a temporary settings
# module object. Rebind its function globals to the process-wide module used by
# the rest of the test suite.
AnswerGenerator.generate_answer.__globals__["config"] = sys.modules[
    "memrewire.core.settings"
].settings


class _FakeCompletions:
    def __init__(self):
        self.kwargs = None

    def create(self, **kwargs):
        self.kwargs = kwargs
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"ok": true}'))],
            usage=SimpleNamespace(
                prompt_tokens=120,
                completion_tokens=10,
                total_tokens=130,
                prompt_tokens_details=SimpleNamespace(
                    cached_tokens=80,
                    cache_write_tokens=20,
                ),
                completion_tokens_details=SimpleNamespace(reasoning_tokens=3),
            ),
        )


class _AnswerClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []
        self.model = "qwen3.5-9b"
        self.enable_thinking = False
        self.prompt_counts = []

    def chat_completion(self, messages, **kwargs):
        self.calls.append((messages, kwargs))
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response

    @staticmethod
    def extract_json(response):
        return json.loads(response)

    @contextmanager
    def usage_scope(self, **context):
        yield

    def count_prompt_tokens(self, messages, template_kwargs=None):
        cells = messages[1]["content"].count("CELL")
        self.prompt_counts.append(cells)
        return 1700 if cells <= 1 else 2100


def test_local_qwen_disables_thinking_via_vllm_chat_template_kwargs():
    client = LLMClient(
        api_key="test",
        model="qwen3-8b",
        base_url="http://127.0.0.1:28000/v1",
        enable_thinking=False,
        use_streaming=False,
    )
    completions = _FakeCompletions()
    client.client = SimpleNamespace(
        chat=SimpleNamespace(completions=completions),
    )

    client.chat_completion(
        [{"role": "user", "content": "Return JSON."}],
        response_format={"type": "json_object"},
        stage="answer",
    )

    assert completions.kwargs["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }

    event = client._usage_events[0]
    assert event["prompt_tokens"] == 120
    assert event["cached_prompt_tokens"] == 80
    assert event["cache_write_prompt_tokens"] == 20
    assert event["cached_prompt_detail_calls"] == 1
    assert event["cache_write_prompt_detail_calls"] == 1
    assert event["uncached_prompt_tokens"] == 40
    assert event["uncached_plus_completion_tokens"] == 50
    assert event["reasoning_tokens"] == 3

    summary = client.usage_summary(question_count=1)
    assert summary["candidate_online_token_cost"]["total_tokens"] == 130
    assert summary["candidate_online_uncached_plus_completion"]["uncached_plus_completion_tokens"] == 50
    assert summary["candidate_online_average_uncached_plus_completion_per_question"] == 50


def test_usage_values_keeps_raw_and_uncached_views_when_cache_details_are_missing():
    values = LLMClient._usage_values({
        "prompt_tokens": 25,
        "completion_tokens": 5,
        "total_tokens": 30,
    })

    assert values["cached_prompt_tokens"] == 0
    assert values["cache_write_prompt_tokens"] == 0
    assert values["cached_prompt_detail_calls"] == 0
    assert values["cache_write_prompt_detail_calls"] == 0
    assert values["uncached_prompt_tokens"] == 25
    assert values["uncached_plus_completion_tokens"] == 30


def test_answer_generation_fails_after_one_transport_attempt():
    client = _AnswerClient([TimeoutError("local server unavailable")])
    generator = AnswerGenerator(client)
    source = type(
        "Source",
        (),
        {"lossless_restatement": "Alice met Bob.", "timestamp": None, "source_turn_ids": []},
    )()

    answer = generator.generate_answer("Who met Bob?", [source])

    assert answer.startswith("Failed to generate answer:")
    assert len(client.calls) == 1
    assert all(call[1]["max_retries"] == 1 for call in client.calls)


def test_answer_generation_does_not_retry_invalid_json():
    client = _AnswerClient(["not-json"])
    generator = AnswerGenerator(client)
    source = type(
        "Source",
        (),
        {"lossless_restatement": "Alice met Bob.", "timestamp": None, "source_turn_ids": []},
    )()

    assert generator.generate_answer("Who met Bob?", [source]) == "not-json"
    assert len(client.calls) == 1


def test_full_prompt_budget_prunes_cells_before_request(monkeypatch):
    settings = sys.modules["memrewire.core.settings"].settings
    monkeypatch.setitem(
        AnswerGenerator.generate_answer.__globals__,
        "QueryContext",
        query_context_module.QueryContext,
    )
    monkeypatch.setattr(settings, "ANSWER_MAX_INPUT_TOKENS", 1800, raising=False)
    monkeypatch.setattr(settings, "CONTEXT_MIN_VISIBLE_CELLS", 1, raising=False)
    monkeypatch.setattr(settings, "USE_JSON_FORMAT", True, raising=False)
    monkeypatch.setattr(
        AnswerGenerator,
        "_format_query_context",
        classmethod(lambda cls, query_context, query="": " ".join(query_context.evidence)),
    )

    client = _AnswerClient(['{"answer":"Alice"}'])
    query_context = query_context_module.QueryContext()
    query_context.evidence = ["CELL highest", "CELL middle", "CELL lowest"]
    query_context.question_type = "general"

    answer = AnswerGenerator(client).generate_answer("Who met Bob?", query_context)

    assert answer == "Alice"
    assert query_context.evidence == ["CELL highest"]
    assert client.prompt_counts == [3, 2, 1]
    assert len(client.calls) == 1


def test_shard_merge_conservatively_handles_legacy_usage_summaries():
    summary = merge_token_summaries(
        [{"by_stage": {"answer": {
            "calls": 1,
            "prompt_tokens": 100,
            "completion_tokens": 10,
            "total_tokens": 110,
        }}}],
        question_count=1,
    )

    assert summary["candidate_online_token_cost"]["total_tokens"] == 110
    assert summary["candidate_online_uncached_plus_completion"]["uncached_plus_completion_tokens"] == 110
    assert summary["candidate_online_uncached_plus_completion"]["cached_prompt_detail_calls"] == 0
