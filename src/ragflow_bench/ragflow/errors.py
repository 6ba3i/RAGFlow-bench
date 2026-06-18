from __future__ import annotations


class RagflowError(Exception):
    pass


class RagflowConfigError(RagflowError):
    pass


class RagflowAPIError(RagflowError):
    def __init__(self, message: str, *, status_code: int | None = None, code: int | None = None, url: str | None = None, raw_body: str | None = None):
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.url = url
        self.raw_body = raw_body
