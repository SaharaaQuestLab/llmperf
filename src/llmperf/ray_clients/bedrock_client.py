import json
import os
import time
from typing import Any, Dict

import ray
from transformers import LlamaTokenizerFast

from llmperf.ray_llm_client import LLMClient
from llmperf.models import RequestConfig
from llmperf import common_metrics
from llmperf.aws_client import AWSSession


class BedrockColdStartClient():
    """Client for OpenAI Chat Completions API."""

    def __init__(self):
        role_arn = os.environ.get('AWS_ROLE_ARN', '')
        if not role_arn:
            raise ValueError("AWS_ROLE_ARN must be set to get access to BedrockClient")
        self.session = AWSSession(role_arn, "exec-layer-bedrock-session", custom_region_name='us-east-1')
        self.cold_start = True

    def measure_cold_start(self, prompt, model) -> float:
        run_time_client = self.session.client("bedrock-runtime")

        message = json.dumps({
            "prompt": prompt
        })

        start_time = time.monotonic()
        # measure cold start
        while self.cold_start:
            print("BedrockColdStartClient: measuring cold start ...")
            try:
                run_time_client.invoke_model_with_response_stream(
                    modelId=model,
                    body=message
                )
                self.cold_start = False
                cold_start_duration = time.monotonic() - start_time
                print(f"BedrockClient: cold start duration: {cold_start_duration}")
                return cold_start_duration
            except run_time_client.exceptions.ModelNotReadyException:
                print("BedrockColdStartClient: waiting for model to start up and sleep for 60s ...")
                time.sleep(60)

@ray.remote
class BedrockClient(LLMClient):
    """Client for OpenAI Chat Completions API."""

    def __init__(self):
        # Sagemaker doesn't return the number of tokens that are generated so we approximate it by
        # using the llama tokenizer.
        self.tokenizer = LlamaTokenizerFast.from_pretrained(
            "hf-internal-testing/llama-tokenizer"
        )
        role_arn = os.environ.get('AWS_ROLE_ARN', '')
        if not role_arn:
            raise ValueError("AWS_ROLE_ARN must be set to get access to BedrockClient")
        self.session = AWSSession(role_arn, "exec-layer-bedrock-session", custom_region_name='us-east-1')

    def llm_request(self, request_config: RequestConfig) -> Dict[str, Any]:
        prompt = request_config.prompt
        prompt, prompt_len = prompt

        model = request_config.model

        run_time_client = self.session.client("bedrock-runtime")

        sampling_params = request_config.sampling_params

        if "max_tokens" in sampling_params:
            sampling_params["max_tokens_to_sample"] = sampling_params["max_tokens"]
            del sampling_params["max_tokens"]


        message = json.dumps({
            "prompt": prompt,
            **request_config.sampling_params
        })

        time_to_next_token = []
        tokens_received = 0
        ttft = 0
        error_response_code = None
        generated_text = ""
        error_msg = ""
        output_throughput = 0
        total_request_time = 0
        metrics = {}

        metrics[common_metrics.ERROR_CODE] = None
        metrics[common_metrics.ERROR_MSG] = ""

        start_time = time.monotonic()
        most_recent_received_token_time = time.monotonic()

        try:
            response = run_time_client.invoke_model_with_response_stream(
                modelId=model,
                body=message
            )

            event_stream = response.get('body')
            generated_text = ""
            for event in event_stream:
                if not ttft:
                    ttft = time.monotonic() - start_time
                chunk = event.get('chunk')
                data = json.loads(chunk.get('bytes').decode())['outputs'][0]['text']
                generated_text += data
                time_to_next_token.append(
                    time.monotonic() - most_recent_received_token_time
                )
                most_recent_received_token_time = time.monotonic()
            
            total_request_time = time.monotonic() - start_time
            tokens_received = len(self.tokenizer.encode(generated_text))
            output_throughput = tokens_received / total_request_time

        except Exception as e:
            print(f"Warning Or Error: {e}")
            print(error_response_code)
            error_msg = str(e)
            error_response_code = 500
            metrics[common_metrics.ERROR_MSG] = error_msg
            metrics[common_metrics.ERROR_CODE] = error_response_code

        metrics[common_metrics.INTER_TOKEN_LAT] = sum(time_to_next_token) #This should be same as metrics[common_metrics.E2E_LAT]. Leave it here for now
        metrics[common_metrics.TTFT] = ttft
        metrics[common_metrics.E2E_LAT] = total_request_time
        metrics[common_metrics.REQ_OUTPUT_THROUGHPUT] = output_throughput
        metrics[common_metrics.NUM_TOTAL_TOKENS] = tokens_received + prompt_len
        metrics[common_metrics.NUM_OUTPUT_TOKENS] = tokens_received
        metrics[common_metrics.NUM_INPUT_TOKENS] = prompt_len

        return metrics, generated_text, request_config

