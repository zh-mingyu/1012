# MemRewire

MemRewire organizes long interaction histories as source-faithful memory cells
connected by a sparse graph. At query time, it routes to relevant cells,
expands a bounded graph neighborhood, and selects up to `K` original source
turns by query similarity for answer generation.

## Repository layout

```text
config.example.py       Example local configuration
requirements.txt        Runtime and benchmark dependencies
merge_memrewire_shards.py Merge sharded evaluation outputs
src/memrewire/            Memory construction, graph, retrieval, and readout
src/memrewire_app.py      High-level system API
src/evaluate_locomo.py  LoCoMo evaluation runner
src/run_paired_graph_comparison.py
                        Paired graph comparison experiment runner
src/locomo_evidence.py  LoCoMo evidence ID normalization
src/metrics.py          Evaluation result summarization
tests/                  Focused unit tests
```

## Setup

Use Python 3.10 or newer. Install the dependencies and expose `src` on the
Python path:

```bash
python -m pip install -r requirements.txt
python -m pip install pytest
export PYTHONPATH=src
```

The embedding model is loaded locally through `sentence-transformers`. Set
`OPENAI_API_KEY`, `OPENAI_BASE_URL`, and `LLM_MODEL` in the environment for an
OpenAI-compatible answer endpoint. Local model paths and other settings can be
placed in a private `config.py` based on `config.example.py`; do not commit
credentials or private paths.

## Run tests

```bash
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src python -m pytest -q
```

## Minimal usage

```python
from memrewire_app import MemRewireSystem
from memrewire.core.models.memory_entry import Dialogue

system = MemRewireSystem(clear_db=True)
system.add_dialogues([
    Dialogue(dialogue_id=1, speaker="Alice", content="The review is Friday."),
    Dialogue(dialogue_id=2, speaker="Bob", content="I will bring the draft."),
])
system.finalize()
answer = system.ask("When is the review?")
print(answer)
```

## LoCoMo evaluation

Pass the dataset and output paths to the evaluator:

```bash
PYTHONPATH=src python src/evaluate_locomo.py \
  --dataset /path/to/locomo10.json \
  --output-dir /path/to/results
```

The dataset, model weights, credentials, and generated results are not included
in this repository.
