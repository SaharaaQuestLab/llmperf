## vllm

### env set up
```bash
export OPENAI_API_BASE=http://34.126.91.78/v1
```

### check nvidia gpu
```bash
nvidia-smi
```


### docker permission issue
```bash
# Run the following command to create the docker group:
sudo groupadd docker

# Add Your User to the docker Group, Replace <username> with your username (or use $USER for the current user):
sudo usermod -aG docker $USER

# To verify that your user is now part of the docker group, use:
groups $USER

# apply changes by either relog in or use newgrp command
newgrp docker
```

### commands ran to generate benchmark for non cc

#### example - Llama-3-8B-Instruct
```bash
sudo docker run --runtime nvidia  --gpus "device=0" --rm --name Meta-Llama-3-8B-Instruct -v /home/huggingface:/root/.cache/huggingface  -d --env HUGGING_FACE_HUB_TOKEN=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  -p 80:8000  --ipc=host  vllm/vllm-openai:v0.4.1  --model meta-llama/Meta-Llama-3-8B-Instruct

curl http://localhost:80/v1/chat/completions   -H "Content-Type: application/json"   -d '{
    "model": "meta-llama/Meta-Llama-3-8B-Instruct",
    "messages": [{"role": "user", "content": "San Francisco is a"}],
    "max_tokens": 7,
    "temperature": 0,
    "stream": true
}'

python token_benchmark_ray.py \
--model "meta-llama/Meta-Llama-3-8B-Instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs_h100_80GB_cc_off" \
--llm-api openai \
--additional-sampling-params '{"temperature": 0}'

```

#### Llama-3.1-8B-Instruct
```bash
docker run --runtime nvidia --gpus "device=0" --rm --name Llama-3.1-8B-Instruct -v /data0/huggingface:/root/.cache/huggingface -d --env HUGGING_FACE_HUB_TOKEN=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 -d -p 80:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B-Instruct

docker run -it --runtime nvidia --gpus "device=0" --rm --name Llama-3.1-8B-Instruct -v /data0/huggingface:/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN=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 -d -p 80:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B-Instruct

# 500 requests
python token_benchmark_ray.py \
--model "meta-llama/Llama-3.1-8B-Instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 500 \
--timeout 600 \
--num-concurrent-requests 1 \
--results-dir "result_outputs_h100_80GB_cc_off" \
--llm-api openai \
--additional-sampling-params '{"temperature": 0}'
```

#### Phi-3-medium-128k-instruct
```bash
docker run -d --runtime nvidia --gpus "device=0" --rm --name Phi-3-medium-128k-instruct -v /data0/huggingface:/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN=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 -p 80:8000 --ipc=host vllm/vllm-openai:v0.5.3 --model microsoft/Phi-3-medium-128k-instruct --gpu-memory-utilization 0.9  --max-model-len 100000

# 200 requests
python token_benchmark_ray.py \
--model "microsoft/Phi-3-medium-128k-instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 200 \
--timeout 1000 \
--num-concurrent-requests 1 \
--results-dir "result_outputs_h100_80GB_cc_off" \
--llm-api openai \
--additional-sampling-params '{"temperature": 0}'

# 500 requests
python token_benchmark_ray.py \
--model "microsoft/Phi-3-medium-128k-instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 500 \
--timeout 3000 \
--num-concurrent-requests 1 \
--results-dir "result_outputs_h100_80GB_cc_off" \
--llm-api openai \
--additional-sampling-params '{"temperature": 0}'
```


#### Llama-3.1-70B-Instruct
```bash
docker run -d  --runtime nvidia --gpus "device=0" --rm --name  Meta-Llama-3.1-70B-Instruct-AWQ-INT4 -v /data0/huggingface:/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN=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 -p 80:8000 --ipc=host vllm/vllm-openai:v0.5.4 --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --gpu-memory-utilization 0.9  --max-model-len 100000

# 200 requests
python token_benchmark_ray.py \
--model "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 200 \
--timeout 5000 \
--num-concurrent-requests 1 \
--results-dir "result_outputs_h100_80GB_cc_off" \
--llm-api openai \
--additional-sampling-params '{"temperature": 0}'
```

### commands ran to generate benchmark for tdx cc

#### Llama-3.1-8B-Instruct
```bash
docker run -d --runtime nvidia --gpus "device=0" --rm --name Llama-3.1-8B-Instruct -v /data0/huggingface:/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN=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 -p 80:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B-Instruct

docker run -it --runtime nvidia --gpus "device=0" --rm --name Llama-3.1-8B-Instruct -v /data0/huggingface:/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN=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 -p 80:8000 --ipc=host vllm/vllm-openai:latest --model meta-llama/Llama-3.1-8B-Instruct

# 500 requests
python token_benchmark_ray.py \
--model "meta-llama/Llama-3.1-8B-Instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 500 \
--timeout 1000 \
--num-concurrent-requests 1 \
--results-dir "result_outputs_h100_80GB_cc_on" \
--llm-api openai \
--additional-sampling-params '{"temperature": 0}'
```

#### Phi-3-medium-128k-instruct
```bash
docker run -d --runtime nvidia --gpus "device=0" --rm --name Phi-3-medium-128k-instruct -v /data0/huggingface:/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN=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 -p 80:8000 --ipc=host vllm/vllm-openai:v0.5.3 --model microsoft/Phi-3-medium-128k-instruct --gpu-memory-utilization 0.9  --max-model-len 100000

# 200 requests
python token_benchmark_ray.py \
--model "microsoft/Phi-3-medium-128k-instruct" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 200 \
--timeout 1000 \
--num-concurrent-requests 1 \
--results-dir "result_outputs_h100_80GB_cc_on" \
--llm-api openai \
--additional-sampling-params '{"temperature": 0}'
```


#### Llama-3.1-70B-Instruct
```bash
docker run -d  --runtime nvidia --gpus "device=0" --rm --name  Meta-Llama-3.1-70B-Instruct-AWQ-INT4 -v /data0/huggingface:/root/.cache/huggingface --env HUGGING_FACE_HUB_TOKEN=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 -p 80:8000 --ipc=host vllm/vllm-openai:v0.5.4 --model hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4 --gpu-memory-utilization 0.9  --max-model-len 100000

# 200 requests
python token_benchmark_ray.py \
--model "hugging-quants/Meta-Llama-3.1-70B-Instruct-AWQ-INT4" \
--mean-input-tokens 550 \
--stddev-input-tokens 150 \
--mean-output-tokens 150 \
--stddev-output-tokens 10 \
--max-num-completed-requests 200 \
--timeout 5000 \
--num-concurrent-requests 1 \
--results-dir "result_outputs_h100_80GB_cc_on" \
--llm-api openai \
--additional-sampling-params '{"temperature": 0}'
```
