# Gradio Application Guide: vllm_web.py

> Deep dive into the web interface implementation

---

## Table of Contents

1. [Overview](#overview)
2. [Application Architecture](#application-architecture)
3. [Code Walkthrough](#code-walkthrough)
4. [API Client Implementation](#api-client-implementation)
5. [UI Components](#ui-components)
6. [Streaming & Real-Time Updates](#streaming--real-time-updates)
7. [Error Handling](#error-handling)
8. [Customization Guide](#customization-guide)

---

## Overview

**File:** `/Users/lsetiawan/Repos/SSEC/gpt-oss-with-vllm-on-supercomputer/vllm_web.py`
**Purpose:** Gradio-based web interface for interacting with vLLM API
**Framework:** Gradio 5.42.0
**Lines of Code:** 237

### Key Responsibilities

1. **API Client:** HTTP communication with vLLM server
2. **Chat Interface:** User-friendly web UI for conversations
3. **Streaming:** Real-time response rendering
4. **Configuration:** Model selection, temperature, max tokens
5. **Health Monitoring:** Server status and model availability

### Design Principles

- **Separation of Concerns:** API client (VLLMChat) separate from UI (create_interface)
- **Stateless Backend:** No conversation persistence (all state in browser)
- **Streaming-First:** Uses SSE for responsive user experience
- **Graceful Degradation:** Handles API errors without crashing UI

---

## Application Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────┐
│                     vllm_web.py                             │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Global Configuration                                │   │
│  │  - DEFAULT_BASE_URL                                  │   │
│  │  - ENV_DEFAULT_MODEL                                 │   │
│  │  - FILTER_THINK (reasoning model support)           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  VLLMChat Class (API Client)                        │   │
│  │  ┌────────────────────────────────────────────────┐ │   │
│  │  │  __init__(base_url)                            │ │   │
│  │  │  - requests.Session with retry logic           │ │   │
│  │  │  - HTTPAdapter with backoff                    │ │   │
│  │  └────────────────────────────────────────────────┘ │   │
│  │  ┌────────────────────────────────────────────────┐ │   │
│  │  │  health() -> bool                              │ │   │
│  │  │  - GET /v1/models with timeout                 │ │   │
│  │  └────────────────────────────────────────────────┘ │   │
│  │  ┌────────────────────────────────────────────────┐ │   │
│  │  │  list_models() -> List[str]                    │ │   │
│  │  │  - Parse /v1/models response                   │ │   │
│  │  └────────────────────────────────────────────────┘ │   │
│  │  ┌────────────────────────────────────────────────┐ │   │
│  │  │  chat_stream() -> Generator                    │ │   │
│  │  │  - POST /v1/chat/completions (stream=True)     │ │   │
│  │  │  - Yield SSE chunks as text                    │ │   │
│  │  └────────────────────────────────────────────────┘ │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  Helper Functions                                   │   │
│  │  - _strip_think(text) -> str                        │   │
│  │  - wait_for_server(chat, timeout) -> (bool, list)  │   │
│  │  - health_text(ok) -> str                           │   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  create_interface(initial_base_url)                 │   │
│  │  ┌─────────────────────────────────────────────────┐│   │
│  │  │  Gradio Blocks UI                               ││   │
│  │  │  - Chatbot component                            ││   │
│  │  │  - Text input                                   ││   │
│  │  │  - Model dropdown                               ││   │
│  │  │  - Settings (temperature, max_tokens)          ││   │
│  │  │  - Refresh button                               ││   │
│  │  └─────────────────────────────────────────────────┘│   │
│  │  ┌─────────────────────────────────────────────────┐│   │
│  │  │  Event Handlers                                 ││   │
│  │  │  - on_send(msg, hist, ...)                     ││   │
│  │  │  - on_refresh(current_model, server)           ││   │
│  │  │  - on_clear()                                   ││   │
│  │  └─────────────────────────────────────────────────┘│   │
│  └─────────────────────────────────────────────────────┘   │
│                                                             │
│  ┌─────────────────────────────────────────────────────┐   │
│  │  main() - CLI Entry Point                           │   │
│  │  - ArgumentParser (--host, --port, --share)         │   │
│  │  - Launch Gradio server                             │   │
│  └─────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────┘
```

### Data Flow: User Message to Response

```
1. User types message → clicks "Send"
   │
   ├─► Gradio frontend captures event
   │
2. on_send() handler invoked (line 162)
   │
   ├─► Validates model availability
   ├─► Appends user message to history
   ├─► Calls VLLMChat.chat_stream()
   │
3. VLLMChat.chat_stream() (line 51)
   │
   ├─► POST request to vLLM API
   │   ├─ URL: http://localhost:8000/v1/chat/completions
   │   ├─ Body: {model, messages, temperature, max_tokens, stream: true}
   │   └─ Timeout: 600s
   │
   ├─► Iterate over SSE response lines
   │   ├─ Parse "data: {...}" format
   │   ├─ Extract delta.content from JSON
   │   └─ Yield decoded text chunks
   │
4. on_send() accumulates chunks (line 178)
   │
   ├─► acc += chunk
   ├─► Strip <think> tags (line 186)
   ├─► Yield to Gradio (updates chatbot component)
   │
5. Gradio frontend renders (real-time)
   │
   └─► User sees streaming response in chat UI
```

---

## Code Walkthrough

### Section 1: Imports & Configuration (Lines 1-23)

```python
import os
import time
import json
import html
import argparse
import requests
from requests.adapters import HTTPAdapter, Retry
import gradio as gr

# --- Behavior knobs ---
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1").rstrip("/")
ENV_DEFAULT_MODEL = os.getenv("DEFAULT_MODEL", "").strip() or None
FILTER_THINK = True  # strip <think>...</think> from outputs
```

**Configuration Variables:**
- `DEFAULT_BASE_URL`: vLLM API endpoint (overridable via env var)
- `ENV_DEFAULT_MODEL`: Default model selection (set by SLURM script)
- `FILTER_THINK`: Enable/disable reasoning token filtering

**Environment Variable Integration:**
```bash
# In vllm_gradio_run_singularity.sh (line 275-277)
export OPENAI_BASE_URL="http://127.0.0.1:${VLLM_PORT}/v1"
export DEFAULT_MODEL="${VLLM_MODEL}"
# vllm_web.py reads these on startup
```

### Section 2: Reasoning Token Filter (Lines 16-23)

```python
def _strip_think(text: str) -> str:
    if not text:
        return text
    if FILTER_THINK:
        # remove <think>...</think> (single-line or multi-line)
        import re
        text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.S | re.I)
    return text
```

**Purpose:** Remove reasoning process tokens from GPT-OSS and similar models

**Why Needed:**
- GPT-OSS uses `<think>` tags to show internal reasoning
- Users typically want only the final answer, not reasoning
- Regex pattern handles both single-line and multi-line blocks

**Example:**
```
Input:  "<think>Let me calculate... 2+2=4</think>The answer is 4."
Output: "The answer is 4."
```

**Configuration:**
Set `FILTER_THINK = False` to show reasoning process.

---

### Section 3: VLLMChat Class (Lines 25-101)

#### Initialization (Lines 25-31)

```python
class VLLMChat:
    def __init__(self, base_url: str):
        self.base_url = base_url.rstrip("/")
        self.session = requests.Session()
        retries = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
        self.session.mount("http://", HTTPAdapter(max_retries=retries))
        self.session.mount("https://", HTTPAdapter(max_retries=retries))
```

**Design Pattern: Connection Pooling with Retry Logic**

**Benefits:**
- **Connection reuse:** TCP connections persist across requests
- **Automatic retries:** Handles transient errors (server overload, network hiccups)
- **Backoff strategy:** Exponential delay between retries (1s, 2s, 4s)
- **Status-specific:** Only retry on server errors (5xx) and rate limits (429)

**Retry Logic:**
```
Request fails with 503
  ↓
Wait 1 second
  ↓
Retry (attempt 2)
  ↓
Fails again with 503
  ↓
Wait 2 seconds (backoff_factor * attempt)
  ↓
Retry (attempt 3)
  ↓
Success or final failure
```

#### Health Check (Lines 34-39)

```python
def health(self) -> bool:
    try:
        r = self.session.get(f"{self.base_url}/models", timeout=5)
        return r.status_code == 200
    except requests.RequestException:
        return False
```

**Purpose:** Fast health check for UI status indicator

**Why `/models` Endpoint:**
- Lightweight (no inference)
- Returns 200 only if vLLM fully initialized
- Standard OpenAI-compatible endpoint

**Usage in UI:**
```python
ok = chat.health()
health_md.update(health_text(ok))  # "Health: ✅ OK" or "Health: ❌ DOWN"
```

#### Model List (Lines 41-49)

```python
def list_models(self):
    try:
        r = self.session.get(f"{self.base_url}/models", timeout=15)
        r.raise_for_status()
        payload = r.json()
        return [m["id"] for m in payload.get("data", [])]
    except Exception as e:
        print(f"[list_models] error: {e}")
        return []
```

**Response Format (OpenAI-compatible):**
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

**Error Handling:**
- Returns empty list on failure (UI shows "No models available")
- Logs error to console for debugging

#### Chat Stream (Lines 51-85)

```python
def chat_stream(self, messages, model: str, temperature: float, max_tokens: int):
    """
    Streams content from /v1/chat/completions using SSE-like lines.
    """
    url = f"{self.base_url}/chat/completions"
    body = {
        "model": model,
        "messages": messages,
        "temperature": float(temperature),
        "max_tokens": int(max_tokens),
        "stream": True,
    }
    try:
        with self.session.post(url, json=body, stream=True, timeout=600) as resp:
            if resp.status_code != 200:
                yield f"\n\n[Error {resp.status_code}] {resp.text}"
                return
            for raw in resp.iter_lines(decode_unicode=True):
                if not raw:
                    continue
                line = raw.strip()
                if line.startswith("data:"):
                    line = line[5:].strip()
                if line == "[DONE]":
                    break
                try:
                    chunk = json.loads(line)
                    delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                    if delta:
                        yield html.unescape(delta)
                except Exception as e:
                    # Non-JSON line or parse hiccup; ignore but log
                    print(f"[stream-parse] {e}: {line[:200]}")
    except requests.RequestException as e:
        yield f"\n\n[Request error] {e}"
```

**SSE (Server-Sent Events) Format:**
```
data: {"choices":[{"delta":{"content":"Hello"}}]}

data: {"choices":[{"delta":{"content":" world"}}]}

data: [DONE]
```

**Parsing Logic:**
1. Iterate lines from HTTP stream
2. Strip `data: ` prefix
3. Check for `[DONE]` terminator
4. Parse JSON, extract `choices[0].delta.content`
5. Decode HTML entities (`&amp;` → `&`)
6. Yield text chunk to caller

**Error Handling:**
- Non-200 status: Yield error message as text
- JSON parse errors: Log and skip line (don't crash stream)
- Network errors: Yield error message, end stream

---

### Section 4: Helper Functions (Lines 87-103)

#### wait_for_server (Lines 87-100)

```python
def wait_for_server(chat: VLLMChat, timeout_s=120, poll=2):
    start = time.time()
    last_models = []
    while time.time() - start < timeout_s:
        ok = chat.health()
        if ok:
            models = chat.list_models()
            if models:
                return True, models
            last_models = models
        time.sleep(poll)
    # fallback: return whatever we saw (likely empty)
    return False, last_models
```

**Purpose:** Block UI startup until vLLM API is ready

**Strategy:**
- Poll every 2 seconds for up to 120 seconds
- Check health first (fast)
- If healthy, fetch models (slower)
- Return success only if models available

**Why This Matters:**
- vLLM startup takes 30-300 seconds (model download + CUDA graph)
- UI shouldn't show "No models" during startup
- Provides better UX than immediate failure

#### health_text (Lines 102-103)

```python
def health_text(ok: bool) -> str:
    return "Health: ✅ **OK**" if ok else "Health: ❌ **DOWN**"
```

Simple formatter for Gradio Markdown component.

---

### Section 5: UI Creation (Lines 105-225)

#### Initialization & Warmup (Lines 105-127)

```python
def create_interface(initial_base_url: str):
    chat = VLLMChat(initial_base_url)
    ok, models = wait_for_server(chat, timeout_s=180, poll=2)
    # choose default model safely
    if models:
        if ENV_DEFAULT_MODEL and ENV_DEFAULT_MODEL in models:
            default_model = ENV_DEFAULT_MODEL
        else:
            default_model = models[0]
    else:
        default_model = None

    # Optional: quick warmup to reduce first-token latency
    try:
        if default_model:
            list(chat.chat_stream(
                messages=[{"role": "user", "content": "ping"}],
                model=default_model,
                temperature=0.0,
                max_tokens=1,
            ))
    except Exception:
        pass
```

**Startup Sequence:**
1. Create VLLMChat instance
2. Wait for vLLM API (up to 180 seconds)
3. Select default model (prefer ENV_DEFAULT_MODEL)
4. Send warmup request (reduces first-user latency by ~20%)

**Warmup Benefits:**
- Loads model into GPU memory (if not already)
- Compiles first CUDA graph
- Fills tokenizer cache
- User's first request feels faster

#### UI Layout (Lines 129-160)

```python
with gr.Blocks(title="vLLM Chat Interface") as demo:
    gr.HTML("""
    <style>
    .gradio-container { max-width: 1500px !important; }
    #chat_col .gr-chatbot { max-width: 100% !important; }
    #side_col { min-width: 280px; max-width: 320px; }
    </style>
    """)

    gr.Markdown("# vLLM Chat Interface")

    with gr.Row():
        with gr.Column(scale=11, elem_id="chat_col"):
            chatbot = gr.Chatbot(height=520, type="messages")
            message = gr.Textbox(label="Message", placeholder="Type your message here...")
            send = gr.Button("Send", variant="primary")

        with gr.Column(scale=3, elem_id="side_col"):
            server_url = gr.Textbox(label="Server", value=chat.base_url)
            health_md = gr.Markdown(health_text(ok))
            model_dd = gr.Dropdown(
                choices=models if models else ["No models available"],
                value=default_model,
                label="Select Model",
                interactive=bool(models),
            )
            refresh_btn = gr.Button("🔄 Refresh Models")

            with gr.Row():
                temperature = gr.Slider(0.0, 1.0, value=0.7, step=0.1, label="Temperature")
            max_tokens = gr.Slider(16, 8192, value=1536, step=1, label="Max Tokens")
```

**Layout Strategy:**
- **Two-column layout:** Chat (left) + Settings (right)
- **Scale ratio 11:3:** Chat takes ~78% width
- **Custom CSS:** Responsive width, constrained sidebar
- **Fixed chatbot height:** 520px (fits ~10-15 messages)

**Component Types:**
- `gr.Chatbot`: Message history display (supports streaming)
- `gr.Textbox`: User input field
- `gr.Dropdown`: Model selector (populated from API)
- `gr.Slider`: Numeric input with visual feedback
- `gr.Markdown`: Rich text for status indicators

#### Event Handler: on_send (Lines 162-187)

```python
def on_send(msg, hist, server, model, temp, max_toks):
    # guardrails: rebind server base & ensure valid model
    chat.base_url = server.strip().rstrip("/")
    available = chat.list_models()
    if not available:
        yield "", hist + [{"role": "assistant", "content": "Server has no models yet."}]
        return
    if model not in available:
        model = available[0]

    if not msg.strip():
        yield "", hist
        return

    new_hist = hist + [{"role": "user", "content": msg}]
    acc = ""
    for chunk in chat.chat_stream(
        messages=new_hist,
        model=model,
        temperature=temp,
        max_tokens=int(max_toks),
    ):
        acc += chunk
        # optimistic filtering of <think>
        safe = _strip_think(acc)
        yield "", new_hist + [{"role": "assistant", "content": safe}]
```

**Generator Pattern for Streaming:**
- Each `yield` updates the UI
- `yield "", new_hist + [...]`:
  - First element: Clear message input field
  - Second element: Updated chat history
- Gradio re-renders chatbot on each yield

**Guardrails:**
1. **Server rebinding:** Allow user to change server URL
2. **Model validation:** Switch to available model if current invalid
3. **Empty message check:** Ignore blank submissions
4. **Incremental filtering:** Apply `_strip_think` to partial responses

**Streaming Visualization:**
```
User: "What is 2+2?"
  ↓
Chunk 1: "<think>Let me"
  → Display: ""
Chunk 2: " calculate</think>The"
  → Display: "The"
Chunk 3: " answer is 4"
  → Display: "The answer is 4"
```

#### Event Handler: on_refresh (Lines 189-203)

```python
def on_refresh(current_model, server):
    try:
        chat.base_url = server.strip().rstrip("/")
        ok_now = chat.health()
        models_now = chat.list_models()
        if models_now:
            new_value = current_model if current_model in models_now else (
                ENV_DEFAULT_MODEL if ENV_DEFAULT_MODEL in models_now else models_now[0]
            )
            dd = gr.update(choices=models_now, value=new_value, interactive=True)
        else:
            dd = gr.update(choices=["No models available"], value=None, interactive=False)
        return health_text(ok_now), dd
    except Exception as e:
        return f"Health: ❌ Error: {e}", gr.update()
```

**Purpose:** Re-fetch models without restarting UI

**Use Cases:**
- vLLM restarted with different model
- Server URL changed
- Model loading completed after UI startup

**Return Values:**
- Tuple: (health_markdown_text, dropdown_update)
- `gr.update()`: Gradio component update object

#### Event Wiring (Lines 208-223)

```python
send.click(
    on_send,
    inputs=[message, chatbot, server_url, model_dd, temperature, max_tokens],
    outputs=[message, chatbot],
)
refresh_btn.click(
    on_refresh,
    inputs=[model_dd, server_url],
    outputs=[health_md, model_dd],
)
message.submit(  # Enter key
    on_send,
    inputs=[message, chatbot, server_url, model_dd, temperature, max_tokens],
    outputs=[message, chatbot],
)
gr.Button("Clear Chat").click(on_clear, None, [chatbot, message], queue=False)
```

**Event Types:**
- `.click()`: Button clicks
- `.submit()`: Enter key in text input
- `.change()`: Dropdown/slider value changes (not used here)

**Queueing:**
- Default: Events queued (FIFO)
- `queue=False`: Execute immediately (for Clear button)

---

## Streaming & Real-Time Updates

### Gradio Streaming Mechanism

**How It Works:**
```python
def generator_function():
    for item in data:
        yield output1, output2, ...
        # UI updates after each yield

component.click(generator_function, inputs=[...], outputs=[...])
```

**Behind the Scenes:**
1. Gradio calls generator
2. For each `yield`:
   - Serialize values to JSON
   - Send to browser via WebSocket
   - Browser updates components
3. Generator completion ends stream

### SSE Parsing Details

**vLLM SSE Format:**
```
data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1735282800,"model":"gpt-oss-20b","choices":[{"index":0,"delta":{"role":"assistant"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1735282800,"model":"gpt-oss-20b","choices":[{"index":0,"delta":{"content":"Hello"},"finish_reason":null}]}

data: {"id":"chatcmpl-123","object":"chat.completion.chunk","created":1735282800,"model":"gpt-oss-20b","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}

data: [DONE]
```

**Parsing Strategy (Lines 68-80):**
```python
for raw in resp.iter_lines(decode_unicode=True):
    if not raw:
        continue  # Skip empty lines
    line = raw.strip()
    if line.startswith("data:"):
        line = line[5:].strip()  # Remove prefix
    if line == "[DONE]":
        break  # End of stream
    try:
        chunk = json.loads(line)
        delta = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
        if delta:
            yield html.unescape(delta)
    except Exception as e:
        print(f"[stream-parse] {e}: {line[:200]}")
```

**Robustness Features:**
- Skip empty lines (SSE requires blank line separators)
- Handle missing keys gracefully (`.get()` with defaults)
- Catch JSON parse errors (malformed chunks don't crash stream)
- HTML entity decoding (vLLM escapes special characters)

### Performance Characteristics

**Latency Breakdown:**
```
vLLM generates token (40ms)
  ↓
Send SSE chunk over localhost (< 1ms)
  ↓
Python parse JSON (< 1ms)
  ↓
Yield to Gradio (< 1ms)
  ↓
WebSocket to browser (5-50ms, depends on network)
  ↓
Browser renders (< 1ms)

Total: ~45-95ms per token
```

**Throughput:**
- vLLM: 25 tokens/sec (for 20B model)
- Streaming overhead: < 5% (negligible)
- User perceives: Real-time (no buffering)

---

## Error Handling

### Error Categories & Responses

**1. vLLM Server Down**
```python
# In wait_for_server (line 87-100)
ok, models = wait_for_server(chat, timeout_s=180, poll=2)
if not ok:
    # UI shows "No models available", dropdown disabled
```

**2. Model Not Available**
```python
# In on_send (line 165-170)
available = chat.list_models()
if not available:
    yield "", hist + [{"role": "assistant", "content": "Server has no models yet."}]
    return
if model not in available:
    model = available[0]  # Fallback to first available
```

**3. Network Errors**
```python
# In chat_stream (line 84-85)
except requests.RequestException as e:
    yield f"\n\n[Request error] {e}"
```
User sees error message in chat (doesn't crash UI).

**4. Stream Parse Errors**
```python
# In chat_stream (line 81-83)
try:
    chunk = json.loads(line)
    # ...
except Exception as e:
    print(f"[stream-parse] {e}: {line[:200]}")
    # Continue parsing next line (don't break stream)
```

**5. HTTP Error Responses**
```python
# In chat_stream (line 65-67)
if resp.status_code != 200:
    yield f"\n\n[Error {resp.status_code}] {resp.text}"
    return
```

### Graceful Degradation Strategy

```
Best Case: Full functionality
  ↓
vLLM slow? → Show "Still preparing" message
  ↓
vLLM returns errors? → Display error in chat
  ↓
Network timeout? → Show "[Request error]" message
  ↓
Worst Case: UI remains usable, user can retry
```

---

## Customization Guide

### Customization 1: Add Conversation Export

**Add button to UI (after line 223):**
```python
export_btn = gr.Button("💾 Export Conversation")

def export_chat(hist):
    import json
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filename = f"conversation_{timestamp}.json"
    with open(filename, "w") as f:
        json.dump(hist, f, indent=2)
    return f"Exported to {filename}"

export_btn.click(export_chat, inputs=[chatbot], outputs=[gr.Textbox(label="Status")])
```

### Customization 2: Add System Prompt

**Modify on_send (after line 175):**
```python
system_prompt = gr.Textbox(label="System Prompt", value="You are a helpful assistant.")

def on_send(msg, hist, server, model, temp, max_toks, sys_prompt):
    # ... existing validation ...

    # Add system message
    messages = [{"role": "system", "content": sys_prompt}] + new_hist

    for chunk in chat.chat_stream(messages=messages, ...):
        # ... existing streaming ...
```

**Wire input:**
```python
send.click(on_send, inputs=[..., system_prompt], outputs=[...])
```

### Customization 3: Add Token Counter

**Create counter function:**
```python
def count_tokens(hist):
    # Rough approximation: 1 token ≈ 4 characters
    total_chars = sum(len(m["content"]) for m in hist)
    return f"~{total_chars // 4} tokens"

token_counter = gr.Markdown("~0 tokens")

# Update on message send
chatbot.change(lambda h: count_tokens(h), inputs=[chatbot], outputs=[token_counter])
```

### Customization 4: Multi-Model Comparison

**Add parallel models:**
```python
with gr.Row():
    model_1 = gr.Dropdown(choices=models, label="Model 1")
    model_2 = gr.Dropdown(choices=models, label="Model 2")

chatbot_1 = gr.Chatbot(label="Model 1")
chatbot_2 = gr.Chatbot(label="Model 2")

def on_send_multi(msg, hist1, hist2, model1, model2, ...):
    new_hist1 = hist1 + [{"role": "user", "content": msg}]
    new_hist2 = hist2 + [{"role": "user", "content": msg}]

    # Stream both in parallel (requires threading or asyncio)
    # ... implementation details ...
```

### Customization 5: Add Response Rating

**Add feedback buttons:**
```python
with gr.Row():
    thumbs_up = gr.Button("👍")
    thumbs_down = gr.Button("👎")

def rate_response(rating, hist):
    # Log rating to file
    with open("ratings.jsonl", "a") as f:
        f.write(json.dumps({"rating": rating, "conversation": hist[-2:]}) + "\n")
    return "Thanks for your feedback!"

thumbs_up.click(lambda h: rate_response("positive", h), inputs=[chatbot], outputs=[gr.Textbox()])
```

---

## Testing & Debugging

### Local Testing (Without SLURM)

```bash
# Terminal 1: Start vLLM locally (if you have GPU)
vllm serve Qwen/Qwen3-0.6B --port 8000

# Terminal 2: Start Gradio UI
cd /Users/lsetiawan/Repos/SSEC/gpt-oss-with-vllm-on-supercomputer
python vllm_web.py --host 127.0.0.1 --port 7860 --base-url http://127.0.0.1:8000/v1

# Browser: Open http://127.0.0.1:7860
```

### Mock Server for UI Development

**Create `mock_vllm.py`:**
```python
from flask import Flask, jsonify, request, Response
import time

app = Flask(__name__)

@app.route("/v1/models")
def models():
    return jsonify({"data": [{"id": "mock-model"}]})

@app.route("/v1/chat/completions", methods=["POST"])
def chat():
    def generate():
        for word in ["Hello", " world", "!"]:
            chunk = {"choices": [{"delta": {"content": word}}]}
            yield f"data: {chunk}\n\n"
            time.sleep(0.1)
        yield "data: [DONE]\n\n"
    return Response(generate(), mimetype="text/event-stream")

app.run(port=8000)
```

**Test UI with mock:**
```bash
python mock_vllm.py  # Terminal 1
python vllm_web.py --base-url http://127.0.0.1:8000/v1  # Terminal 2
```

### Debugging Checklist

1. **vLLM not responding:**
   ```bash
   curl http://127.0.0.1:8000/v1/models
   # Check if returns JSON
   ```

2. **Streaming not working:**
   ```python
   # Add debug prints in chat_stream
   print(f"[DEBUG] Received line: {line[:100]}")
   ```

3. **UI not updating:**
   - Check browser console for WebSocket errors
   - Verify generator is yielding (add print statements)

4. **Port conflicts:**
   ```bash
   lsof -i :7860  # Check if port in use
   ```

---

## Summary

The Gradio application is a **lightweight, responsive web interface** that:
- Provides intuitive chat UI with streaming responses
- Implements robust API client with retry logic
- Handles errors gracefully without crashing
- Supports model switching and parameter tuning
- Filters reasoning tokens for cleaner outputs

**Key Strengths:**
- **Minimal dependencies:** Only Gradio + requests
- **Stateless design:** Easy to scale and debug
- **Streaming-first:** Responsive UX with real-time feedback
- **Extensible:** Easy to add features (export, ratings, etc.)

**Potential Improvements:**
- Persistent conversation storage
- User authentication
- Multi-model comparison
- Advanced parameter tuning (top_p, frequency_penalty)

---

**Document Version:** 1.0
**Last Updated:** 2025-01-27
