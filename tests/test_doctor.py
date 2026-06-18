import yaml

from ragflow_bench import cli


class _FakeDoctorClient:
    created_embedding_models: list[str | None] = []

    def __init__(self, connection):
        self.base_url = connection.resolved_base_url()

    def health_check(self):
        return {"ok": True}

    def system_status(self):
        return {"status": "ok"}

    def list_models(self):
        return []

    def create_dataset(self, **kwargs):
        self.__class__.created_embedding_models.append(kwargs.get("embedding_model"))
        return {"id": "doctor-ds", "embedding_model": kwargs.get("embedding_model")}

    def upload_document(self, dataset_id, path):
        return []


def test_doctor_warns_when_new_dataset_embedding_model_missing(tmp_path, monkeypatch, capsys):
    config_path = tmp_path / "doctor.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "kind": "custom",
                    "mode": "smoke",
                    "custom": {
                        "corpus_dir": ".",
                        "questions_path": "tests/test_doctor.py",
                    },
                },
                "dataset": {
                    "strategy": "create_new_dataset",
                    "name": "doctor-dataset",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "RagflowClient", _FakeDoctorClient)

    cli.doctor(config=str(config_path))

    captured = capsys.readouterr()
    assert "dataset.embedding_model not set" in captured.out


def test_doctor_uses_requested_embedding_model_for_probe_dataset(tmp_path, monkeypatch):
    _FakeDoctorClient.created_embedding_models = []
    config_path = tmp_path / "doctor.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "benchmark": {
                    "kind": "custom",
                    "mode": "smoke",
                    "custom": {
                        "corpus_dir": ".",
                        "questions_path": "tests/test_doctor.py",
                    },
                },
                "dataset": {
                    "strategy": "create_new_dataset",
                    "name": "doctor-dataset",
                    "embedding_model": "bge-m3",
                },
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli, "RagflowClient", _FakeDoctorClient)

    cli.doctor(config=str(config_path))

    assert _FakeDoctorClient.created_embedding_models == ["bge-m3"]


def test_main_callback_loads_dotenv_before_commands(tmp_path, monkeypatch):
    env_path = tmp_path / ".env"
    env_path.write_text("RAGFLOW_API_KEY=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("RAGFLOW_API_KEY", raising=False)

    cli.main_callback(verbose=False)

    assert cli.os.environ["RAGFLOW_API_KEY"] == "from-dotenv"
