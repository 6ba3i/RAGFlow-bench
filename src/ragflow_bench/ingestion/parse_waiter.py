from __future__ import annotations

from ragflow_bench.ragflow.client import RagflowClient


def wait_for_parse(client: RagflowClient, dataset_id: str, document_ids: list[str], *, poll_interval: float = 2.0, timeout: float = 600.0):
    return client.wait_for_documents_parsed(dataset_id, document_ids, poll_interval=poll_interval, timeout=timeout)
