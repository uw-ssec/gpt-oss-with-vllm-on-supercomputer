# API Reference: vLLM OpenAI-Compatible Endpoints

> Complete documentation for vLLM server API endpoints

---

## Table of Contents

1. [Overview](#overview)
2. [Base URL & Authentication](#base-url--authentication)
3. [Endpoints](#endpoints)
4. [Models API](#models-api)
5. [Chat Completions API](#chat-completions-api)
6. [Responses API (GPT-OSS Specific)](#responses-api-gpt-oss-specific)
7. [Error Handling](#error-handling)
8. [Rate Limits & Performance](#rate-limits--performance)
9. [Examples](#examples)

---

## Overview

The vLLM server provides an **OpenAI-compatible REST API** for language model inference. This enables drop-in replacement for OpenAI's API in existing applications.

**Compatibility:**
- OpenAI Python SDK: ✅ Supported
- OpenAI Node.js SDK: ✅ Supported
- LangChain: ✅ Supported
- curl / HTTP clients: ✅ Supported

**Base Container:** `vllm/vllm-openai:gptoss`
**Default Port:** 8000
**Protocol:** HTTP (HTTPS if configured separately)

---

## Base URL & Authentication

### Base URL

```
http://<compute-node>:<port>/v1
```

**Examples:**
- Local (on compute node): `http://127.0.0.1:8000/v1`
- Via SSH tunnel: `http://localhost:8000/v1`
- Direct (if accessible): `http://gpu05.cluster.edu:8000/v1`

### Authentication

**Default: No authentication required**

The vLLM server runs on isolated compute nodes without public access. Authentication is handled at the HPC cluster level (SSH, SLURM).

**For Production:**
Consider adding authentication via:
- Reverse proxy (nginx with basic auth)
- API gateway (Kong, Traefik)
- Network firewall rules

**Environment Variables:**
```bash
# Client-side (for OpenAI SDK compatibility)
export OPENAI_API_KEY="sk-local"  # Dummy key (not validated)
export OPENAI_BASE_URL="http://localhost:8000/v1"
```

---

## Endpoints

### Endpoint Summary

| Endpoint | Method | Purpose | Streaming |
|----------|--------|---------|-----------|
| `/v1/models` | GET | List available models | No |
| `/v1/chat/completions` | POST | Chat-style generation | Yes |
| `/v1/completions` | POST | Text completion | Yes |
| `/v1/responses` | POST | GPT-OSS reasoning format | No |
| `/health` | GET | Health check | No |
| `/metrics` | GET | Prometheus metrics | No |

---

## Models API

### List Models

**Endpoint:** `GET /v1/models`

**Description:** Returns list of loaded models in the vLLM server.

#### Request

```bash
curl http://localhost:8000/v1/models
```

#### Response

```json
{
  "object": "list",
  "data": [
    {
      "id": "openai/gpt-oss-20b",
      "object": "model",
      "created": 1735282800,
      "owned_by": "vllm"
    }
  ]
}
```

**Response Fields:**
- `object`: Always `"list"`
- `data`: Array of model objects
  - `id`: Model identifier (HuggingFace model ID)
  - `object`: Always `"model"`
  - `created`: Unix timestamp of model load
  - `owned_by`: Always `"vllm"`

**Notes:**
- vLLM typically serves one model at a time
- Model ID matches the `--model` flag passed to `vllm serve`
- Use this endpoint for health checking (returns 200 only if ready)

---

## Chat Completions API

### Create Chat Completion

**Endpoint:** `POST /v1/chat/completions`

**Description:** Generate model responses in conversational format (OpenAI Chat API compatible).

#### Request

**Headers:**
```
Content-Type: application/json
```

**Body Parameters:**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `model` | string | Yes | - | Model ID from `/v1/models` |
| `messages` | array | Yes | - | Conversation history |
| `temperature` | float | No | 1.0 | Sampling temperature (0.0-2.0) |
| `top_p` | float | No | 1.0 | Nucleus sampling threshold |
| `max_tokens` | integer | No | 16 | Maximum tokens to generate |
| `stream` | boolean | No | false | Enable streaming responses |
| `stop` | array | No | null | Stop sequences |
| `presence_penalty` | float | No | 0.0 | Penalize repeated content |
| `frequency_penalty` | float | No | 0.0 | Penalize token frequency |
| `n` | integer | No | 1 | Number of completions |

**Messages Format:**
```json
{
  "messages": [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "Hello!"},
    {"role": "assistant", "content": "Hi! How can I help?"},
    {"role": "user", "content": "What is 2+2?"}
  ]
}
```

**Roles:**
- `system`: Instructions for the model (not all models support)
- `user`: User messages
- `assistant`: Previous model responses (for multi-turn conversations)

#### Non-Streaming Response

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Say hello"}],
    "max_tokens": 50,
    "temperature": 0.7
  }'
```

**Response:**
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1735282800,
  "model": "openai/gpt-oss-20b",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Hello! How can I assist you today?"
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 10,
    "completion_tokens": 8,
    "total_tokens": 18
  }
}
```

**Response Fields:**
- `id`: Unique completion ID
- `object`: Always `"chat.completion"`
- `created`: Unix timestamp
- `model`: Model used
- `choices`: Array of completions (typically one)
  - `index`: Choice index (0-based)
  - `message`: Generated message
    - `role`: Always `"assistant"`
    - `content`: Generated text
  - `finish_reason`: Why generation stopped
    - `"stop"`: Natural end or stop sequence
    - `"length"`: Hit `max_tokens` limit
    - `"content_filter"`: Content filtered (rare)
- `usage`: Token counts
  - `prompt_tokens`: Input tokens
  - `completion_tokens`: Generated tokens
  - `total_tokens`: Sum of above

#### Streaming Response

```bash
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "messages": [{"role": "user", "content": "Count to 5"}],
    "max_tokens": 50,
    "stream": true
  }'
```

**Response (SSE format):**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1735282800,"model":"openai/gpt-oss-20b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1735282800,"model":"openai/gpt-oss-20b","choices":[{"index":0,"delta":{"content":"1"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1735282800,"model":"openai/gpt-oss-20b","choices":[{"index":0,"delta":{"content":", "},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1735282800,"model":"openai/gpt-oss-20b","choices":[{"index":0,"delta":{"content":"2"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1735282800,"model":"openai/gpt-oss-20b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Streaming Format:**
- Each line starts with `data: `
- Contains JSON chunk object
- `delta` field contains incremental content
- Empty `delta` with `finish_reason` indicates end
- Final line is `data: [DONE]`

**Parsing Streaming Responses:**
```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={"model": "openai/gpt-oss-20b", "messages": [...], "stream": True},
    stream=True
)

for line in response.iter_lines(decode_unicode=True):
    if not line or not line.startswith("data: "):
        continue
    if line == "data: [DONE]":
        break
    data = json.loads(line[6:])  # Skip "data: " prefix
    delta = data["choices"][0]["delta"].get("content", "")
    if delta:
        print(delta, end="", flush=True)
```

---

## Responses API (GPT-OSS Specific)

### Create Response

**Endpoint:** `POST /v1/responses`

**Description:** GPT-OSS reasoning model API (separate reasoning and output).

**Note:** This endpoint is specific to GPT-OSS models. Standard models may not support it.

#### Request

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "input": "What is 2+2?",
    "max_output_tokens": 100
  }'
```

**Body Parameters:**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `model` | string | Yes | Model ID |
| `input` | string | Yes | User prompt |
| `max_output_tokens` | integer | No | Max tokens in output |

#### Response

```json
{
  "id": "resp-123",
  "object": "response",
  "created": 1735282800,
  "model": "openai/gpt-oss-20b",
  "output": [
    {
      "type": "reasoning",
      "content": [
        {
          "type": "text",
          "text": "Let me calculate 2+2. This is basic arithmetic..."
        }
      ]
    },
    {
      "type": "output",
      "content": [
        {
          "type": "text",
          "text": "The answer is 4."
        }
      ]
    }
  ],
  "output_text": "The answer is 4.",
  "usage": {
    "prompt_tokens": 10,
    "output_tokens": 15,
    "total_tokens": 25
  }
}
```

**Response Fields:**
- `output`: Array of reasoning and output blocks
  - `type`: `"reasoning"` or `"output"`
  - `content`: Array of content blocks
    - `type`: `"text"`
    - `text`: Content string
- `output_text`: Final output only (reasoning removed)

**Use Cases:**
- Show reasoning process to users
- Debug model's thought process
- Compare reasoning quality

---

## Error Handling

### Error Response Format

```json
{
  "error": {
    "message": "Invalid request: max_tokens must be positive",
    "type": "invalid_request_error",
    "param": "max_tokens",
    "code": "invalid_value"
  }
}
```

### Common Error Codes

| HTTP Status | Error Type | Cause | Solution |
|-------------|------------|-------|----------|
| 400 | `invalid_request_error` | Malformed request | Check JSON syntax, required fields |
| 404 | `model_not_found` | Model doesn't exist | Use `/v1/models` to list available models |
| 422 | `invalid_request_error` | Invalid parameters | Check parameter types and ranges |
| 500 | `server_error` | Internal error | Check vLLM logs, retry |
| 503 | `service_unavailable` | Server overloaded | Wait and retry |

### Timeout Errors

**Client-side timeout:**
```python
import requests
try:
    response = requests.post(url, json=data, timeout=120)
except requests.Timeout:
    print("Request timed out (server may be busy)")
```

**Recommendations:**
- Set timeout ≥ 60 seconds for long generations
- Implement exponential backoff for retries
- Use streaming to avoid timeouts

---

## Rate Limits & Performance

### Throughput Characteristics

**Single Request (20B model, A100):**
- First token latency: 150-200ms
- Tokens per second: 25-30 tok/s
- Max context: 2K-40K tokens (model dependent)

**Concurrent Requests:**
- Max concurrent: 8-64 (depends on `--max-num-seqs`)
- Continuous batching: Requests processed in parallel
- Queue: FIFO (first in, first out)

### Best Practices

**1. Use Streaming for Long Responses:**
```json
{"stream": true}  // User sees progressive output
```

**2. Set Reasonable `max_tokens`:**
```json
{"max_tokens": 512}  // Don't use 4096 if only need short answer
```

**3. Use Stop Sequences:**
```json
{"stop": ["\n\n", "User:", "</think>"]}  // Reduce over-generation
```

**4. Batch Similar Requests:**
- vLLM efficiently batches concurrent requests
- Submit multiple requests simultaneously for better GPU utilization

**5. Monitor Queue Length:**
```bash
curl http://localhost:8000/metrics | grep vllm:num_requests_waiting
```

---

## Examples

### Example 1: Simple Chat (Python)

```python
import os
import requests

os.environ["OPENAI_BASE_URL"] = "http://localhost:8000/v1"

response = requests.post(
    f"{os.environ['OPENAI_BASE_URL']}/chat/completions",
    json={
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 50
    }
)

print(response.json()["choices"][0]["message"]["content"])
```

### Example 2: Streaming Chat (Python)

```python
import requests
import json

response = requests.post(
    "http://localhost:8000/v1/chat/completions",
    json={
        "model": "openai/gpt-oss-20b",
        "messages": [{"role": "user", "content": "Write a poem"}],
        "max_tokens": 200,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines(decode_unicode=True):
    if line.startswith("data: ") and line != "data: [DONE]":
        data = json.loads(line[6:])
        delta = data["choices"][0]["delta"].get("content", "")
        if delta:
            print(delta, end="", flush=True)
print()  # Newline at end
```

### Example 3: OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-local"  # Dummy key
)

response = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=50
)

print(response.choices[0].message.content)
```

### Example 4: Streaming with OpenAI SDK

```python
from openai import OpenAI

client = OpenAI(
    base_url="http://localhost:8000/v1",
    api_key="sk-local"
)

stream = client.chat.completions.create(
    model="openai/gpt-oss-20b",
    messages=[{"role": "user", "content": "Count to 10"}],
    max_tokens=100,
    stream=True
)

for chunk in stream:
    if chunk.choices[0].delta.content:
        print(chunk.choices[0].delta.content, end="", flush=True)
print()
```

### Example 5: Multi-Turn Conversation

```python
import requests

base_url = "http://localhost:8000/v1"
conversation = []

def chat(user_message):
    conversation.append({"role": "user", "content": user_message})

    response = requests.post(
        f"{base_url}/chat/completions",
        json={
            "model": "openai/gpt-oss-20b",
            "messages": conversation,
            "max_tokens": 150
        }
    )

    assistant_message = response.json()["choices"][0]["message"]["content"]
    conversation.append({"role": "assistant", "content": assistant_message})
    return assistant_message

# Multi-turn conversation
print(chat("Hi, I'm Alice"))
print(chat("What's my name?"))  # Model should remember "Alice"
```

### Example 6: GPT-OSS Reasoning API

```bash
curl http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{
    "model": "openai/gpt-oss-20b",
    "input": "Solve: If x + 5 = 12, what is x?",
    "max_output_tokens": 200
  }' | jq .
```

**Parse output:**
```bash
# Extract just the final answer (skip reasoning)
curl -sS http://localhost:8000/v1/responses \
  -H "Content-Type: application/json" \
  -d '{...}' | jq -r '.output_text'
```

### Example 7: Batch Processing (Shell)

```bash
# process_queries.sh
while IFS= read -r query; do
  echo "Processing: $query"
  curl -sS http://localhost:8000/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
      "model": "openai/gpt-oss-20b",
      "messages": [{"role": "user", "content": "'"$query"'"}],
      "max_tokens": 100
    }' | jq -r '.choices[0].message.content'
  echo "---"
done < queries.txt
```

---

## Summary

The vLLM OpenAI-compatible API provides:
- **Standard Endpoints:** Drop-in replacement for OpenAI API
- **Streaming Support:** Real-time response generation
- **Reasoning Models:** GPT-OSS `/v1/responses` endpoint
- **High Performance:** vLLM optimizations (paged attention, continuous batching)
- **Easy Integration:** Works with OpenAI SDKs, LangChain, etc.

**Key Differences from OpenAI:**
- No authentication by default (HPC environment)
- Single model at a time (not multi-model)
- GPT-OSS specific `/v1/responses` endpoint
- Local deployment (no cloud API limits)

**Best Practices:**
- Use streaming for long responses
- Set reasonable `max_tokens` limits
- Implement retry logic with backoff
- Monitor GPU memory and queue length
- Use stop sequences to reduce over-generation

---

**Document Version:** 1.0
**Last Updated:** 2025-01-27
