from pathlib import Path

import yaml

from ragflow_bench import wizard as wizard_module


def test_wizard_writes_embedding_model_for_new_dataset(tmp_path, monkeypatch):
    answers = {
        "Which benchmark?": "frames",
        "Which mode?": "smoke",
        "How many questions?": "default smoke limit",
        "RAGFlow base URL": "http://127.0.0.1:80",
        "API key environment variable name": "RAGFLOW_API_KEY",
        "LLM ID environment variable name": "RAGFLOW_LLM_ID",
        "Dataset strategy": "create_new_dataset",
        "New dataset name": "frames-smoke",
        "Dataset embedding model": "bge-m3",
        "Delimiter": "\\n",
        "Similarity threshold": "0.05",
        "Vector similarity weight": "0.3",
        "Temperature": "0.0",
        "Top p": "0.1",
        "Output folder": "outputs/frames_smoke_<timestamp>",
        "FRAMES artifact directory": "data/frames-smoke",
        "Config path": str(tmp_path / "wizard.yaml"),
    }
    int_answers = {
        "Chunk size": 512,
        "Retrieval top_k": 128,
        "Retrieval page_size": 20,
        "Chat top_n": 8,
        "Max tokens": 128,
    }
    confirm_answers = {
        "Fresh session per question?": True,
        "Quote references?": True,
        "Refine multiturn?": False,
        "Run immediately?": False,
    }

    monkeypatch.setattr(
        wizard_module.Prompt,
        "ask",
        staticmethod(lambda prompt, **kwargs: answers[prompt]),
    )
    monkeypatch.setattr(
        wizard_module.IntPrompt,
        "ask",
        staticmethod(lambda prompt, **kwargs: int_answers[prompt]),
    )
    monkeypatch.setattr(
        wizard_module.Confirm,
        "ask",
        staticmethod(lambda prompt, **kwargs: confirm_answers[prompt]),
    )

    target, should_run = wizard_module.run_wizard()

    assert target == Path(answers["Config path"])
    assert should_run is False
    payload = yaml.safe_load(target.read_text(encoding="utf-8"))
    assert payload["dataset"]["embedding_model"] == "bge-m3"
    assert payload["benchmark"]["frames"]["mapping_path"] == "data/frames-smoke/frames_mapping.json"
    assert payload["benchmark"]["frames"]["local_corpus_dir"] == "data/frames-smoke/corpus"
