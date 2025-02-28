from typing import Any, Dict, List, Optional, Tuple
from pydantic import BaseModel


class RequestConfig(BaseModel):
    """The configuration for a request to the LLM API.

    Args:
        model: The model to use.
        prompt: The prompt to provide to the LLM API.
        sampling_params: Additional sampling parameters to send with the request.
            For more information see the Router app's documentation for the completions
        llm_api: The name of the LLM API to send the request to.
        metadata: Additional metadata to attach to the request for logging or validation purposes.
        additional_headers: Additional headers to send with the request.
        request_timeout: The time out for each request to the llm api, default is 180 seconds.
    """

    model: str
    prompt: Tuple[str, int]
    sampling_params: Optional[Dict[str, Any]] = None
    llm_api: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None
    additional_headers: Optional[Dict[str, Any]] = None
    request_timeout: Optional[int] = 180
