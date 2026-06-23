from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import requests

from ragflow_bench.config import RagflowConnectionConfig
from ragflow_bench.logging_utils import redact_text
from ragflow_bench.ragflow.errors import RagflowAPIError, RagflowConfigError


class RagflowClient:
    def __init__(self, connection: RagflowConnectionConfig, timeout: int = 180):
        self.connection = connection
        self.base_url = connection.resolved_base_url().rstrip("/")
        self.api_key = connection.resolved_api_key()
        self.timeout = timeout
        self.session = requests.Session()
        self.last_response: requests.Response | None = None
        if not self.base_url:
            raise RagflowConfigError("RAGFlow base URL is required")

    def _headers(self, require_auth: bool = True) -> dict[str, str]:
        headers = {"Accept": "application/json"}
        if require_auth:
            if not self.api_key:
                raise RagflowConfigError("RAGFlow API key is required")
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, method: str, path: str, *, require_auth: bool = True, **kwargs: Any) -> Any:
        url = f"{self.base_url}{path}"
        headers = self._headers(require_auth=require_auth)
        if "headers" in kwargs:
            headers.update(kwargs.pop("headers"))
        response = self.session.request(method, url, headers=headers, timeout=self.timeout, **kwargs)
        self.last_response = response
        content_type = response.headers.get("content-type", "")
        parsed: Any = None
        if "application/json" in content_type:
            parsed = response.json()
        else:
            parsed = response.text
        if response.status_code >= 400:
            raise RagflowAPIError(
                f"HTTP {response.status_code} for {path}",
                status_code=response.status_code,
                url=url,
                raw_body=redact_text(response.text, [self.api_key or ""]),
            )
        if isinstance(parsed, dict) and parsed.get("code", 0) not in (0, None):
            raise RagflowAPIError(
                parsed.get("message") or f"RAGFlow returned code {parsed.get('code')}",
                status_code=response.status_code,
                code=parsed.get("code"),
                url=url,
                raw_body=redact_text(json.dumps(parsed, ensure_ascii=False), [self.api_key or ""]),
            )
        return parsed

    def health_check(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/system/healthz", require_auth=False)

    def system_status(self) -> dict[str, Any]:
        return self._request("GET", "/api/v1/system/status")

    def list_models(self) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/v1/models")
        return payload.get("data", [])

    def create_dataset(self, *, name: str, description: str = "", embedding_model: str | None = None, chunk_method: str = "naive", parser_config: dict[str, Any] | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {"name": name, "description": description, "chunk_method": chunk_method}
        if embedding_model:
            body["embedding_model"] = embedding_model
        if parser_config:
            body["parser_config"] = parser_config
        payload = self._request("POST", "/api/v1/datasets", json=body)
        return payload.get("data", {})

    def list_datasets(self, *, page: int = 1, page_size: int = 100, include_parsing_status: bool = False) -> list[dict[str, Any]]:
        payload = self._request("GET", "/api/v1/datasets", params={"page": page, "page_size": page_size, "include_parsing_status": str(include_parsing_status).lower()})
        return payload.get("data", [])

    def get_dataset(self, dataset_id: str) -> dict[str, Any]:
        payload = self._request("GET", f"/api/v1/datasets/{dataset_id}")
        return payload.get("data", {})

    def upload_document(self, dataset_id: str, path: str | Path) -> list[dict[str, Any]]:
        file_path = Path(path)
        with file_path.open("rb") as handle:
            payload = self._request(
                "POST",
                f"/api/v1/datasets/{dataset_id}/documents",
                files={"file": (file_path.name, handle, "application/octet-stream")},
            )
        return payload.get("data", [])

    def patch_document_metadata(self, dataset_id: str, document_id: str, *, meta_fields: dict[str, Any] | None = None, name: str | None = None, chunk_method: str | None = None, parser_config: dict[str, Any] | None = None, enabled: bool | None = None) -> dict[str, Any]:
        body: dict[str, Any] = {}
        if meta_fields is not None:
            body["meta_fields"] = meta_fields
        if name is not None:
            body["name"] = name
        if chunk_method is not None:
            body["chunk_method"] = chunk_method
        if parser_config is not None:
            body["parser_config"] = parser_config
        if enabled is not None:
            body["enabled"] = enabled
        payload = self._request("PATCH", f"/api/v1/datasets/{dataset_id}/documents/{document_id}", json=body)
        return payload.get("data", {})

    def start_parse(self, dataset_id: str, document_ids: list[str]) -> dict[str, Any]:
        try:
            return self._request("POST", f"/api/v1/datasets/{dataset_id}/documents/parse", json={"document_ids": document_ids})
        except RagflowAPIError as exc:
            if exc.status_code == 404:
                return self._request("POST", f"/api/v1/datasets/{dataset_id}/chunks", json={"document_ids": document_ids})
            raise

    def list_documents(self, dataset_id: str, *, page: int = 1, page_size: int = 100, run: str | None = None) -> dict[str, Any]:
        params: dict[str, Any] = {"page": page, "page_size": page_size}
        if run:
            params["run"] = run
        payload = self._request("GET", f"/api/v1/datasets/{dataset_id}/documents", params=params)
        return payload.get("data", {})

    def wait_for_documents_parsed(self, dataset_id: str, document_ids: list[str], *, poll_interval: float = 2.0, timeout: float = 600.0) -> list[dict[str, Any]]:
        deadline = time.time() + timeout
        latest: list[dict[str, Any]] = []
        while time.time() < deadline:
            listing = self.list_documents(dataset_id, page=1, page_size=max(100, len(document_ids)))
            docs = listing.get("docs", [])
            latest = [doc for doc in docs if doc.get("id") in set(document_ids)]
            if latest and all(doc.get("run") == "DONE" for doc in latest):
                return latest
            if latest and any(doc.get("run") == "FAIL" for doc in latest):
                raise RagflowAPIError(f"One or more documents failed to parse in dataset {dataset_id}", raw_body=json.dumps(latest))
            time.sleep(poll_interval)
        raise RagflowAPIError(f"Timed out waiting for documents to parse in dataset {dataset_id}", raw_body=json.dumps(latest))

    def create_chat(self, *, name: str, dataset_ids: list[str], llm_id: str, prompt_config: dict[str, Any] | None = None, top_n: int = 8, top_k: int = 128, similarity_threshold: float = 0.05, vector_similarity_weight: float = 0.3) -> dict[str, Any]:
        body: dict[str, Any] = {
            "name": name,
            "dataset_ids": dataset_ids,
            "llm_id": llm_id,
            "top_n": top_n,
            "top_k": top_k,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
        }
        if prompt_config is not None:
            body["prompt_config"] = prompt_config
        payload = self._request("POST", "/api/v1/chats", json=body)
        return payload.get("data", {})

    def create_session(self, chat_id: str, *, name: str = "New session") -> dict[str, Any]:
        payload = self._request("POST", f"/api/v1/chats/{chat_id}/sessions", json={"name": name})
        return payload.get("data", {})

    def retrieve(self, *, question: str, dataset_ids: list[str] | None = None, document_ids: list[str] | None = None, page_size: int = 20, similarity_threshold: float = 0.05, vector_similarity_weight: float = 0.3, top_k: int = 128) -> dict[str, Any]:
        body: dict[str, Any] = {
            "question": question,
            "page_size": page_size,
            "similarity_threshold": similarity_threshold,
            "vector_similarity_weight": vector_similarity_weight,
            "top_k": top_k,
        }
        if dataset_ids:
            body["dataset_ids"] = dataset_ids
        if document_ids:
            body["document_ids"] = document_ids
        payload = self._request("POST", "/api/v1/retrieval", json=body)
        return payload.get("data", {})

    def ask_chat(self, *, question: str, chat_id: str, session_id: str | None = None, llm_id: str | None = None, quote: bool = True, refine_multiturn: bool = False, stream: bool = False, max_tokens: int = 128, temperature: float = 0.0, top_p: float = 0.1) -> dict[str, Any]:
        body: dict[str, Any] = {
            "chat_id": chat_id,
            "question": question,
            "stream": stream,
            "quote": quote,
            "refine_multiturn": refine_multiturn,
            "max_tokens": max_tokens,
            "temperature": temperature,
            "top_p": top_p,
        }
        if session_id:
            body["session_id"] = session_id
        if llm_id:
            body["llm_id"] = llm_id
        payload = self._request("POST", "/api/v1/chat/completions", json=body)
        return payload.get("data", {})
