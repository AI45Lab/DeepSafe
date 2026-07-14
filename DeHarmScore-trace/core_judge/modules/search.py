from __future__ import annotations

import hashlib
import json
import logging
import sys
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote

import requests

from ..schemas import EvidenceItem, SearchProviderConfig, SearchSettings, SearchTrace


logger = logging.getLogger(__name__)

MAX_SEARCH_SNIPPET_CHARS = 800
MAX_SEARCH_TITLE_CHARS = 160


class WebSearchClient:
    """Multi-provider web search client (Jina / Serper) with file cache.

    Simplified from the original `JinaSearchClient`: no proxy injection, no
    reliance on project-specific proxy utilities. Falls back through all
    configured providers on error.
    """

    def __init__(self, settings: SearchSettings) -> None:
        self.settings = settings
        self._sessions: dict[str, requests.Session] = {}
        self._rate_limiters: dict[str, _RateLimiter] = {}
        for provider in settings.providers:
            session = requests.Session()
            self._sessions[provider.provider] = session
            self._rate_limiters[provider.provider] = _RateLimiter(provider.rate_limit_rps)
        self._artifact_dir = Path(settings.artifact_dir)
        self._cache = _FileCache(
            cache_dir=Path(settings.cache_dir) if settings.cache_dir else None,
            ttl_seconds=settings.cache_ttl_seconds,
        )

    def search(self, queries: list[str]) -> list[SearchTrace]:
        if not self.settings.enabled or not self.settings.providers:
            return []
        self._artifact_dir.mkdir(parents=True, exist_ok=True)
        traces: list[SearchTrace] = []
        for query_index, query in enumerate(_merge_unique(queries), start=1):
            payload = self._cache.get_or_load(
                json.dumps(
                    {
                        "kind": "search",
                        "providers": [
                            {"provider": p.provider, "endpoint": p.endpoint}
                            for p in self.settings.providers
                        ],
                        "top_k": self.settings.top_k,
                        "query": query,
                    },
                    ensure_ascii=False,
                    sort_keys=True,
                ),
                lambda query=query: self._fetch_provider_payloads(query),
            )
            results = _normalize_search_results(
                payload,
                query=query,
                query_index=query_index,
                top_k=self.settings.top_k,
            )
            artifact_path = self._write_artifact(query=query, results=results)
            traces.append(
                SearchTrace(
                    query=query,
                    artifact_path=str(artifact_path),
                    results=results,
                )
            )
        print(f"[search] {self._cache.summary()}", file=sys.stderr, flush=True)
        return traces

    def _fetch_provider_payloads(self, query: str) -> dict[str, Any]:
        last_error: Exception | None = None
        for provider in self.settings.providers:
            try:
                return self._fetch_provider(provider, query)
            except (requests.RequestException, ValueError) as exc:
                last_error = exc
                logger.debug(
                    "Search failed provider=%s query=%r: %s: %s",
                    provider.provider, query, type(exc).__name__, exc,
                )
        if last_error is not None:
            raise last_error
        return {"provider": "", "payload": {}}

    def _fetch_provider(self, provider: SearchProviderConfig, query: str) -> dict[str, Any]:
        if provider.provider == "serper":
            return self._fetch_serper(provider, query)
        if provider.provider == "jina":
            return self._fetch_jina(provider, query)
        raise ValueError(f"Unsupported search provider: {provider.provider}")

    def _fetch_jina(self, provider: SearchProviderConfig, query: str) -> dict[str, Any]:
        headers = {"Accept": "application/json"}
        api_key = provider.resolve_api_key()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        self._rate_limiters["jina"].acquire()
        response = self._sessions["jina"].get(
            provider.endpoint.rstrip("/") + "/" + quote(query, safe=""),
            headers=headers,
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        try:
            return {"provider": "jina", "payload": response.json()}
        except ValueError:
            return {"provider": "jina", "raw_text": response.text}

    def _fetch_serper(self, provider: SearchProviderConfig, query: str) -> dict[str, Any]:
        api_key = provider.resolve_api_key()
        if not api_key:
            raise ValueError("Missing serper api key.")
        self._rate_limiters["serper"].acquire()
        response = self._sessions["serper"].post(
            provider.endpoint,
            json={"q": query, "num": self.settings.top_k},
            headers={"X-API-KEY": api_key, "Content-Type": "application/json"},
            timeout=self.settings.request_timeout_seconds,
        )
        response.raise_for_status()
        return {"provider": "serper", "payload": response.json()}

    def _write_artifact(self, *, query: str, results: list[EvidenceItem]) -> Path:
        key = hashlib.sha256(
            json.dumps(
                {"query": query, "results": [item.to_dict() for item in results]},
                ensure_ascii=False,
                sort_keys=True,
            ).encode("utf-8")
        ).hexdigest()[:16]
        artifact_path = self._artifact_dir / f"search_{key}.json"
        artifact_path.write_text(
            json.dumps(
                {"query": query, "results": [item.to_dict() for item in results]},
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
        return artifact_path


def _normalize_search_results(
    raw_payload: dict[str, Any],
    *,
    query: str,
    query_index: int,
    top_k: int,
) -> list[EvidenceItem]:
    provider = str(raw_payload.get("provider", "jina")).strip().lower() or "jina"
    payload = raw_payload.get("payload")
    if isinstance(payload, dict):
        items = payload.get("organic", []) if provider == "serper" else payload.get("data", payload.get("results", []))
        if isinstance(items, list):
            normalized: list[EvidenceItem] = []
            seen_urls: set[str] = set()
            for item_index, item in enumerate(items[:top_k], start=1):
                if not isinstance(item, dict):
                    continue
                url = str(item.get("url", item.get("link", ""))).strip()
                dedupe_key = url or json.dumps(item, ensure_ascii=False, sort_keys=True)
                if dedupe_key in seen_urls:
                    continue
                seen_urls.add(dedupe_key)
                content = str(item.get("content", item.get("snippet", item.get("description", "")))).strip()
                normalized.append(
                    EvidenceItem(
                        id=f"s{query_index}_{item_index}",
                        source_kind="search",
                        query=query,
                        title=_truncate(str(item.get("title", "")).strip(), MAX_SEARCH_TITLE_CHARS),
                        text=_truncate(content, MAX_SEARCH_SNIPPET_CHARS),
                        url=url,
                        source_name="serper_search" if provider == "serper" else "jina_search",
                    )
                )
            return normalized
    raw_text = str(raw_payload.get("raw_text", "")).strip()
    if not raw_text:
        return []
    return [
        EvidenceItem(
            id=f"s{query_index}_1",
            source_kind="search",
            query=query,
            title="",
            text=_truncate(raw_text, MAX_SEARCH_SNIPPET_CHARS),
            url="",
            source_name="serper_search" if provider == "serper" else "jina_search",
        )
    ]


def _truncate(text: str, max_chars: int) -> str:
    normalized = " ".join(text.split())
    if len(normalized) <= max_chars:
        return normalized
    return normalized[: max_chars - 3].rstrip() + "..."


def _merge_unique(values: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        text = " ".join(str(value).split()).strip()
        key = text.casefold()
        if not text or key in seen:
            continue
        seen.add(key)
        out.append(text)
    return out


class _RateLimiter:
    def __init__(self, rps: float) -> None:
        self._min_interval = 1.0 / rps if rps and rps > 0 else 0.0
        self._lock = threading.Lock()
        self._next_allowed = 0.0

    def acquire(self) -> None:
        if self._min_interval <= 0:
            return
        with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait <= 0:
                self._next_allowed = now + self._min_interval
                wait = 0.0
            else:
                self._next_allowed = self._next_allowed + self._min_interval
        if wait > 0:
            time.sleep(wait)


class _FileCache:
    def __init__(self, *, cache_dir: Path | None, ttl_seconds: int) -> None:
        self.cache_dir = cache_dir
        self.ttl_seconds = max(0, ttl_seconds)
        self.cache_hits = 0
        self.cache_misses = 0

    def get_or_load(self, key: str, loader: Callable[[], Any]) -> Any:
        if self.cache_dir is None:
            self.cache_misses += 1
            return loader()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()}.json"
        if cache_path.exists():
            try:
                cached = json.loads(cache_path.read_text(encoding="utf-8"))
                cached_at = float(cached.get("cached_at", 0.0))
                if self.ttl_seconds == 0 or time.time() - cached_at <= self.ttl_seconds:
                    self.cache_hits += 1
                    return cached.get("payload")
            except (ValueError, TypeError, json.JSONDecodeError):
                logger.debug("Ignoring invalid search cache %s", cache_path)
        self.cache_misses += 1
        payload = loader()
        cache_path.write_text(
            json.dumps({"cached_at": time.time(), "payload": payload}, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        return payload

    def summary(self) -> str:
        total = self.cache_hits + self.cache_misses
        if total == 0:
            return "no cache operations"
        if self.cache_hits == total:
            return f"all {total} queries from cache"
        if self.cache_misses == total:
            return f"all {total} queries fetched fresh"
        return f"{self.cache_hits} cache hits, {self.cache_misses} fresh fetches"


# Back-compat alias; code elsewhere may still reference `JinaSearchClient`.
JinaSearchClient = WebSearchClient
