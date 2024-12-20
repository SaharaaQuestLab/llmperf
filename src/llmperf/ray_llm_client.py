import abc
from typing import Any, Dict, Tuple

from llmperf.models import RequestConfig


class ColdStartClient:
    """A client for making requests to a LLM API e.g Anyscale Endpoints."""

    @abc.abstractmethod
    def measure_cold_start(self, prompt, model) -> float:
        """
            Measure cold start duration for the llm-api
        Returns:
            cold start duration in sec
        """
        ...

class LLMClient:
    """A client for making requests to a LLM API e.g Anyscale Endpoints."""

    @abc.abstractmethod
    def llm_request(
        self, request_config: RequestConfig
    ) -> Tuple[Dict[str, Any], str, RequestConfig]:
        """Make a single completion request to a LLM API

        Returns:
            Metrics about the performance charateristics of the request.
            The text generated by the request to the LLM API.
            The request_config used to make the request. This is mainly for logging purposes.

        """
        ...
