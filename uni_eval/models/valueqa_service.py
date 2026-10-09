"""Choose a configured remote API or a local OpenAI-compatible model service."""

from urllib.parse import urlparse

from uni_eval.models.api import APIModel
from uni_eval.registry import MODELS


def select_service(connection, api=None, local=None):
    if connection not in ("api", "local"):
        raise ValueError("Set model.connection to api or local; no connection/model is selected by default")
    selected = api if connection == "api" else local
    if not isinstance(selected, dict):
        raise ValueError("Configure model." + connection)
    settings = dict(selected)
    for field in ("model_name", "api_base"):
        if not isinstance(settings.get(field), str) or not settings[field].strip():
            raise ValueError("Set model." + connection + "." + field)
    address = urlparse(settings["api_base"])
    if address.scheme not in ("http", "https") or not address.hostname:
        raise ValueError("model." + connection + ".api_base must be an HTTP(S) service address")
    if settings.get("mode", "chat") != "chat":
        raise ValueError("ValueQA requires an OpenAI-compatible chat/completions service")
    if settings.get("max_retries", 0) != 0:
        raise ValueError("Set max_retries=0; the framework manages and records retries")
    settings.setdefault("max_retries", 0)
    settings.setdefault("api_key", "ENV" if connection == "api" else "EMPTY")
    settings.setdefault("concurrency", 10 if connection == "api" else 1)
    if not isinstance(settings["concurrency"], int) or settings["concurrency"] < 1:
        raise ValueError("Model concurrency must be a positive integer")
    return settings


@MODELS.register_module()
class ValueQAServiceModel(APIModel):
    def __init__(self, connection, api=None, local=None):
        settings = select_service(connection, api, local)
        self.connection = connection
        super().__init__(**settings)
