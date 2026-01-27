# SLURM Script Deep Dive: vllm_gradio_run_singularity.sh

> Comprehensive guide to understanding and modifying the orchestration script

---

## Table of Contents

1. [Overview](#overview)
2. [Script Structure](#script-structure)
3. [Section-by-Section Analysis](#section-by-section-analysis)
4. [Configuration & Customization](#configuration--customization)
5. [Monitoring & Health Checks](#monitoring--health-checks)
6. [Error Handling & Cleanup](#error-handling--cleanup)
7. [Common Modifications](#common-modifications)

---

## Overview

**File:** `/Users/lsetiawan/Repos/SSEC/gpt-oss-with-vllm-on-supercomputer/vllm_gradio_run_singularity.sh`
**Purpose:** Orchestrate vLLM and Gradio services on SLURM-managed GPU compute nodes
**Execution Context:** Runs on compute node after SLURM allocates resources

### Responsibilities

1. Environment configuration (modules, conda, cache directories)
2. Service lifecycle management (start, monitor, stop)
3. Health checking and readiness detection
4. Progress reporting for long-running operations
5. Cleanup on exit (graceful and forced)
6. SSH tunnel command generation for users

---

## Script Structure

### File Organization (374 lines)

```
Lines 1-18:   SLURM directives (#SBATCH)
Lines 19-40:  Environment variables and defaults
Lines 41-63:  CLI argument parsing
Lines 64-81:  Directory setup and logging configuration
Lines 82-98:  HuggingFace cache management
Lines 99-110: Cleanup trap function
Lines 111-128: Info banner and port forwarding file
Lines 129-173: Python environment detection
Lines 174-179: Clean stale processes/logs
Lines 180-208: vLLM server launch (Singularity)
Lines 209-262: vLLM readiness detection with progress
Lines 263-310: Gradio UI launch and readiness
Lines 311-325: Success summary
Lines 326-373: Continuous monitoring loop
```

### Control Flow Diagram

```
START
  │
  ├─► Parse CLI arguments (--model, --vllm-port, etc.)
  │
  ├─► Setup environment
  │   ├─► Load SLURM modules (gcc, cuda)
  │   ├─► Activate conda environment
  │   └─► Create cache directories
  │
  ├─► Launch vLLM server (background)
  │   └─► Wait for /v1/models endpoint (polling loop)
  │
  ├─► Launch Gradio UI (background)
  │   └─► Wait for Gradio HTTP server (polling loop)
  │
  ├─► Print success banner + SSH command
  │
  └─► Monitor loop (infinite)
      ├─► Check process health (kill -0 $PID)
      ├─► Report GPU stats (nvidia-smi)
      ├─► Check API responsiveness (curl)
      └─► Sleep 30 seconds
      │
      └─► If process dies: log + break → cleanup trap → EXIT
```

---

## Section-by-Section Analysis

### Section 1: SLURM Directives (Lines 1-18)

```bash
#!/bin/bash
#SBATCH --job-name=vllm_gradio
#SBATCH --comment=pytorch
#SBATCH --partition=amd_a100nv_8  # Active partition
#SBATCH --time=48:00:00           # Max 48 hours
#SBATCH --nodes=1                 # Single node
#SBATCH --ntasks-per-node=1       # One task
#SBATCH --gres=gpu:1              # One GPU (change to gpu:2 for TP=2)
#SBATCH --cpus-per-task=8         # 8 CPU cores
```

**Purpose:** SLURM scheduler configuration

**Key Parameters:**
- `--partition`: Queue name (cluster-specific, multiple options commented out)
- `--time`: Walltime limit (48 hours default, adjust per cluster policy)
- `--gres=gpu:1`: GPU count (must match `TP_SIZE` for tensor parallelism)
- `--cpus-per-task`: CPU cores for tokenization/preprocessing

**Customization Examples:**
```bash
# For multi-GPU model (e.g., 70B)
#SBATCH --gres=gpu:4
# Also set: TP_SIZE=4 in environment variables

# For shorter test runs
#SBATCH --time=1:00:00

# For different partition
#SBATCH --partition=gh200_1  # GH200 GPUs
```

---

### Section 2: Environment Variables (Lines 19-40)

```bash
set +e  # Don't exit on errors (handle manually)

SERVER="$(hostname)"  # Compute node hostname
GRADIO_PORT="${GRADIO_PORT:-7860}"
VLLM_PORT="${VLLM_PORT:-8000}"
VLLM_MODEL="${VLLM_MODEL:-Qwen/Qwen3-0.6B}"
SIF_PATH="${SIF_PATH:-/scratch/$USER/gpt-oss-with-vllm-on-supercomputer/vllm-gptoss.sif}"

MAX_MODEL_LEN="${MAX_MODEL_LEN:-40960}"
TP_SIZE="${TP_SIZE:-1}"
GPU_MEM_UTIL="${GPU_MEM_UTIL:-0.90}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
SCHED_STEPS="${SCHED_STEPS:-1}"

export VLLM_CACHE_ROOT="${VLLM_CACHE_ROOT:-/scratch/${USER}/gpt-oss-with-vllm-on-supercomputer/.vllm}"
export XDG_CACHE_HOME="${XDG_CACHE_HOME:-/scratch/${USER}/.gradio_cache}"
export TMPDIR="${TMPDIR:-/scratch/${USER}/tmp}"
```

**Variable Precedence:**
1. Environment variable (if set before job submission)
2. CLI argument (parsed later)
3. Default value (after `:-`)

**Example Overrides:**
```bash
# Method 1: Export before sbatch
export VLLM_MODEL="openai/gpt-oss-20b"
sbatch vllm_gradio_run_singularity.sh

# Method 2: SLURM --export flag
sbatch --export=ALL,VLLM_MODEL=openai/gpt-oss-20b vllm_gradio_run_singularity.sh

# Method 3: CLI argument (parsed lines 54-63)
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b
```

**Performance Tuning Variables:**
- `MAX_MODEL_LEN`: Context window (trade-off: memory vs. length)
- `GPU_MEM_UTIL`: Reserve 10% for fragmentation (0.90 = safe)
- `MAX_NUM_SEQS`: Batch size (higher = throughput, more memory)
- `TP_SIZE`: Tensor parallel GPUs (must match `--gres`)

---

### Section 3: CLI Argument Parsing (Lines 44-63)

```bash
print_help() {
  cat <<EOF
Usage: $0 [--model <hf_model>] [--vllm-port <port>] [--gradio-port <port>] [--sif </path/to.sif>]

Examples:
  $0 --model openai/gpt-oss-20b
  sbatch --export=ALL,VLLM_MODEL=openai/gpt-oss-20b,VLLM_PORT=9000 $0
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)        VLLM_MODEL="$2"; shift 2;;
    --vllm-port)    VLLM_PORT="$2"; shift 2;;
    --gradio-port)  GRADIO_PORT="$2"; shift 2;;
    --sif)          SIF_PATH="$2"; shift 2;;
    -h|--help)      print_help; exit 0;;
    *) echo "Unknown arg: $1"; print_help; exit 1;;
  esac
done
```

**Purpose:** Override defaults via command-line flags

**Parsing Logic:**
- `shift 2`: Consume flag and value (e.g., `--model openai/gpt-oss-20b`)
- Final precedence: CLI args > env vars > defaults

**Usage Patterns:**
```bash
# Interactive testing
srun --gres=gpu:1 ./vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B --vllm-port 9000

# Batch submission with custom ports
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b --vllm-port 8100 --gradio-port 7100
```

---

### Section 4: Directory Setup (Lines 64-98)

```bash
WORK_DIR="/scratch/$USER/gpt-oss-with-vllm-on-supercomputer"
LOG_DIR="${WORK_DIR}/logs"
mkdir -p "$WORK_DIR" "$LOG_DIR" "$XDG_CACHE_HOME" "$TMPDIR" "${VLLM_CACHE_ROOT}"

JOB_ID="${SLURM_JOB_ID:-none}"
if [ "$JOB_ID" = "none" ]; then
  # Interactive mode (srun)
  VLLM_LOG="${LOG_DIR}/vllm_server.log"
  GRADIO_LOG="${LOG_DIR}/gradio_server.log"
  PORT_FWD_FILE="${WORK_DIR}/port_forwarding.txt"
else
  # Batch mode (sbatch)
  VLLM_LOG="${LOG_DIR}/vllm_server_${JOB_ID}.log"
  GRADIO_LOG="${LOG_DIR}/gradio_server_${JOB_ID}.log"
  PORT_FWD_FILE="${WORK_DIR}/port_forwarding_${JOB_ID}.txt"
fi

# HF_HOME setup (with detection of pre-existing value)
if [ -n "${HF_HOME}" ]; then
  HF_PRESET=1  # User already set it
else
  HF_HOME="/scratch/${USER}/.huggingface"
  HF_PRESET=0  # Script set default
fi
mkdir -p "${HF_HOME}/hub"
```

**Purpose:** Create necessary directories and log files

**Directory Structure:**
```
/scratch/$USER/
├── gpt-oss-with-vllm-on-supercomputer/
│   ├── logs/
│   │   ├── vllm_server_<JOBID>.log      # vLLM startup + inference logs
│   │   └── gradio_server_<JOBID>.log    # Gradio UI logs
│   ├── port_forwarding_<JOBID>.txt      # Generated SSH command
│   ├── vllm-gptoss.sif                  # Singularity container
│   └── .vllm/                           # vLLM compilation cache
├── .huggingface/
│   └── hub/                             # Model weights cache
└── .gradio_cache/                       # Gradio temp files
```

**Interactive vs. Batch:**
- Interactive (`srun`): No job ID, generic log names
- Batch (`sbatch`): Job ID embedded in log names for multi-job tracking

---

### Section 5: Cleanup Trap (Lines 99-110)

```bash
cleanup() {
  echo "[$(date)] Cleaning up…"
  [ -n "$GRADIO_PID" ] && kill -TERM "$GRADIO_PID" 2>/dev/null && sleep 2 && kill -9 "$GRADIO_PID" 2>/dev/null
  [ -n "$VLLM_PID" ]   && kill -TERM "$VLLM_PID"   2>/dev/null && sleep 2 && kill -9 "$VLLM_PID"   2>/dev/null
  [ -n "$WATCH_PID" ]  && kill -TERM "$WATCH_PID"  2>/dev/null || true
  pkill -f "vllm serve" 2>/dev/null || true
  echo "[$(date)] Done."
}
trap cleanup EXIT INT TERM
```

**Purpose:** Ensure services stop gracefully (or forcefully) on exit

**Signal Handling:**
- `EXIT`: Normal script termination
- `INT`: User presses Ctrl+C (SIGINT)
- `TERM`: SLURM job cancellation (scancel)

**Cleanup Strategy:**
1. Send `SIGTERM` (graceful shutdown)
2. Wait 2 seconds for process to exit
3. Send `SIGKILL` (force kill) if still running
4. Fallback: `pkill -f` to catch orphaned processes

**Why This Matters:**
- Prevents zombie processes consuming GPU memory
- Releases SLURM resources properly
- Avoids "port already in use" errors on next run

---

### Section 6: Python Environment Detection (Lines 129-173)

```bash
# Load modules (cluster-specific)
if [ -f /etc/profile.d/modules.sh ]; then . /etc/profile.d/modules.sh; fi
module load gcc/10.2.0 cuda/12.1

# Conda activation (handles multiple install locations)
if command -v conda >/dev/null 2>&1; then
  CONDA_BASE="$(conda info --base 2>/dev/null)"
  if [ -n "$CONDA_BASE" ] && [ -f "$CONDA_BASE/etc/profile.d/conda.sh" ]; then
    . "$CONDA_BASE/etc/profile.d/conda.sh"
    conda activate vllm-hpc 2>/dev/null || true
  fi
else
  # Fallback: common path
  if [ -f "/scratch/${USER}/miniconda3/etc/profile.d/conda.sh" ]; then
    . "/scratch/${USER}/miniconda3/etc/profile.d/conda.sh"
    conda activate vllm-hpc 2>/dev/null || true
  fi
fi

# Validate Python version
PYTHON_BIN="$(command -v python3 || true)"
PY_VER_OK="$($PYTHON_BIN -c 'import sys; print(int(sys.version_info[:2] >= (3,8)))' 2>/dev/null)"

if [ "$PY_VER_OK" != "1" ]; then
  # Try explicit env path
  if [ -x "/scratch/${USER}/miniconda3/envs/vllm-hpc/bin/python3" ]; then
    PYTHON_BIN="/scratch/${USER}/miniconda3/envs/vllm-hpc/bin/python3"
  fi
fi
```

**Purpose:** Locate suitable Python interpreter for Gradio

**Detection Strategy:**
1. Try active conda environment's Python
2. Try system `python3` (if version ≥ 3.8)
3. Fallback to explicit path (`/scratch/$USER/miniconda3/envs/vllm-hpc/bin/python3`)
4. Fail with error if no suitable Python found

**Why Complex Logic?**
- Compute nodes may have different module environments
- Users may install conda in different locations
- Some clusters have outdated system Python (e.g., 3.6)

---

### Section 7: vLLM Launch (Lines 180-208)

```bash
echo "🚀 Starting vLLM server..."
cd "$WORK_DIR"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

nohup singularity exec --nv "$SIF_PATH" \
  vllm serve "${VLLM_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    --dtype auto \
    --tensor-parallel-size "${TP_SIZE}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --num-scheduler-steps "${SCHED_STEPS}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL}" \
    --max-num-seqs "${MAX_NUM_SEQS}" \
    --generation-config vllm \
    --trust-remote-code \
    --use-tqdm-on-load \
    --download-dir "${CACHE_ROOT}" \
    > "$VLLM_LOG" 2>&1 &

VLLM_PID=$!
echo "vLLM PID: $VLLM_PID"
```

**Command Breakdown:**

| Component | Purpose |
|-----------|---------|
| `nohup` | Prevent hangup signal (survives parent exit) |
| `singularity exec --nv` | Run in container with NVIDIA GPU support |
| `vllm serve` | Start OpenAI-compatible API server |
| `--host 0.0.0.0` | Bind to all interfaces (needed for Gradio to connect) |
| `--dtype auto` | Autodetect FP16/BF16 based on GPU |
| `--tensor-parallel-size` | Multi-GPU sharding |
| `--trust-remote-code` | Allow custom model code (needed for some HF models) |
| `--use-tqdm-on-load` | Progress bars in logs for downloads |
| `> "$VLLM_LOG" 2>&1 &` | Redirect output + background |

**Common Modifications:**
```bash
# Add quantization for larger models
--quantization awq  # or bitsandbytes

# Limit max requests
--max-num-seqs 32   # Default is 64

# Enable prefix caching (for repeated prompts)
--enable-prefix-caching
```

---

### Section 8: Progress Watcher (Lines 209-269)

```bash
progress_watch() {
  stdbuf -oL tail -n +0 -F "$VLLM_LOG" 2>/dev/null \
  | awk '!seen[$0]++' \
  | while IFS= read -r line; do
      case "$line" in
        *"Resolved architecture:"*)
          echo "🔧 $(echo "$line" | sed -E 's/.*Resolved architecture: *//')"
          ;;
        *"Loading weights took "*)
          echo "✅ Weights loaded ($(echo "$line" | sed -E 's/.*Loading weights took *//') )"
          ;;
        *"Application startup complete."*)
          echo "✅ vLLM API is ready!"
          break
          ;;
      esac
    done
}

progress_watch &  WATCH_PID=$!

# Friendly "still preparing" pulses
ELAPSED=0
until curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
  sleep 10
  ELAPSED=$((ELAPSED+10))
  echo "⏳ Still preparing vLLM API… (${ELAPSED}s)"
done

kill "$WATCH_PID" 2>/dev/null || true
```

**Purpose:** Provide user-friendly progress updates during long startup

**Components:**
1. **progress_watch function:** Parse vLLM log for milestones
   - Uses `awk '!seen[$0]++'` to deduplicate lines
   - Pattern matching for key events (loading weights, graph capture, etc.)
2. **Background execution:** Runs in parallel with main script
3. **Polling loop:** Check API availability every 10 seconds
4. **User feedback:** Print emoji-prefixed messages to stdout

**Key Milestones Detected:**
- Model architecture resolved
- Weights loaded (with timing)
- CUDA graph captured (with memory usage)
- HTTP server started

**Why This Matters:**
- First model download can take 10-30 minutes (user needs feedback)
- CUDA graph compilation adds 2-5 minutes (looks like hang without messages)
- Helps diagnose issues (e.g., stuck at "loading weights" → network problem)

---

### Section 9: Monitoring Loop (Lines 326-373)

```bash
LAST_HEARTBEAT=$(date +%s)
while true; do
  # Check process health
  if ! kill -0 "$VLLM_PID" 2>/dev/null; then
    echo "[$(date)] ERROR: vLLM process died"
    tail -60 "$VLLM_LOG"
    break
  fi
  if ! kill -0 "$GRADIO_PID" 2>/dev/null; then
    echo "[$(date)] ERROR: Gradio process died"
    tail -60 "$GRADIO_LOG"
    break
  fi

  # Heartbeat every 5 minutes
  NOW=$(date +%s)
  if (( NOW - LAST_HEARTBEAT >= 300 )); then
    echo "[$(date)] 💓 Heartbeat: services running"

    # GPU status
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu,temperature.gpu \
      --format=csv,noheader,nounits | \
    while IFS=',' read -r idx name used total util temp; do
      mem_percent=$(( (used * 100) / (total == 0 ? 1 : total) ))
      printf "  GPU%s (%s): %sMB/%sMB (%s%%) | Util: %s%% | Temp: %s°C\n" \
        "${idx}" "${name}" "${used}" "${total}" "${mem_percent}" "${util}" "${temp}"
    done

    # API health checks
    if curl -s --max-time 5 "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; then
      echo "✅ vLLM API responsive"
    else
      echo "⚠️  vLLM API not responding"
    fi

    LAST_HEARTBEAT=$NOW
  fi
  sleep 30
done
```

**Purpose:** Continuous monitoring until job ends or service crashes

**Monitoring Checks:**
1. **Process liveness:** `kill -0 $PID` (check if process exists)
2. **GPU utilization:** `nvidia-smi` (memory, temperature, compute usage)
3. **API health:** HTTP GET to `/v1/models` (check service responsiveness)

**Heartbeat Frequency:**
- Process checks: Every 30 seconds
- GPU stats: Every 5 minutes
- API health: Every 5 minutes

**Failure Handling:**
- If vLLM dies: Print last 60 lines of log, exit loop → cleanup trap
- If Gradio dies: Print last 60 lines of log, exit loop → cleanup trap
- If API unresponsive: Warn but continue (temporary issue)

---

## Configuration & Customization

### Tuning for Different Models

**Small Models (< 10B parameters):**
```bash
export MAX_MODEL_LEN=8192
export GPU_MEM_UTIL=0.95
export MAX_NUM_SEQS=128
sbatch vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B
```

**Large Models (34B-70B parameters):**
```bash
#SBATCH --gres=gpu:2  # In script header
export TP_SIZE=2
export MAX_MODEL_LEN=2048
export GPU_MEM_UTIL=0.90
export MAX_NUM_SEQS=16
sbatch vllm_gradio_run_singularity.sh --model meta-llama/Llama-2-70b-chat-hf
```

**Long Context Models:**
```bash
export MAX_MODEL_LEN=131072  # 128K context
export GPU_MEM_UTIL=0.85     # Need more headroom
export MAX_NUM_SEQS=4        # Fewer concurrent requests
sbatch vllm_gradio_run_singularity.sh --model gradientai/Llama-3-70B-Instruct-Gradient-1048k
```

### Adding Custom vLLM Flags

**Edit lines 190-204:**
```bash
nohup singularity exec --nv "$SIF_PATH" \
  vllm serve "${VLLM_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    # ... existing flags ...
    --enable-prefix-caching \        # NEW: Cache common prefixes
    --enable-chunked-prefill \       # NEW: Reduce TTFT
    --max-num-batched-tokens 8192 \  # NEW: Limit batch size
    > "$VLLM_LOG" 2>&1 &
```

### Site-Specific Customization

**For Different HPC Clusters:**

1. **Update module loads (lines 133-134):**
   ```bash
   # Example: NCSA Delta cluster
   module load gcc/11.2.0 cuda/11.8.0 openmpi/4.1.2
   ```

2. **Update partition names (lines 4-12):**
   ```bash
   #SBATCH --partition=gpuA100x4  # Cluster-specific name
   ```

3. **Update login node in port forwarding (line 128):**
   ```bash
   echo "ssh ... ${USER}@login.delta.ncsa.illinois.edu" > "$PORT_FWD_FILE"
   ```

---

## Monitoring & Health Checks

### Health Check Methods

**1. Process-Level (kill -0)**
```bash
if ! kill -0 "$VLLM_PID" 2>/dev/null; then
  # Process doesn't exist → crashed or killed
fi
```
- **Pro:** Fast (no network I/O)
- **Con:** Doesn't detect hung processes

**2. API-Level (curl)**
```bash
if curl -s --max-time 5 "http://127.0.0.1:${VLLM_PORT}/v1/models"; then
  # API responding → service healthy
fi
```
- **Pro:** Tests actual functionality
- **Con:** Slower, false positives if API under load

**3. GPU-Level (nvidia-smi)**
```bash
nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits
```
- **Pro:** Detects GPU memory leaks
- **Con:** Doesn't correlate with service health

### Log Analysis

**vLLM Log (`vllm_server_<JOBID>.log`):**
```bash
# Check for errors
grep -i "error\|exception\|failed" $VLLM_LOG

# View startup sequence
grep -E "Loading weights|KV cache|Graph capturing" $VLLM_LOG

# Check request latency
grep "ms/request" $VLLM_LOG
```

**Gradio Log (`gradio_server_<JOBID>.log`):**
```bash
# Check for startup
grep "Running on local URL" $GRADIO_LOG

# View requests
grep "POST /api" $GRADIO_LOG
```

---

## Error Handling & Cleanup

### Cleanup Scenarios

**1. Normal Exit (script completes)**
```
User: scancel <JOBID>
SLURM: Sends SIGTERM to job
Script: trap catches TERM → cleanup() → kill services → exit
```

**2. User Interrupt (Ctrl+C in srun)**
```
User: Presses Ctrl+C
Bash: Sends SIGINT to script
Script: trap catches INT → cleanup() → kill services → exit
```

**3. SLURM Timeout (job exceeds --time)**
```
SLURM: Sends SIGTERM then SIGKILL (after grace period)
Script: trap catches TERM → cleanup() → exit
```

**4. Process Crash (vLLM dies)**
```
vLLM: Crashes (OOM, CUDA error, etc.)
Monitoring loop: Detects via kill -0 → break → exit
Script: trap catches EXIT → cleanup() → kill remaining services
```

### Defensive Practices

**1. Check before cleanup:**
```bash
[ -n "$GRADIO_PID" ] && kill -TERM "$GRADIO_PID"
# Don't try to kill if PID is empty
```

**2. Fallback cleanup:**
```bash
pkill -f "vllm serve" 2>/dev/null || true
# Catch orphaned processes if PID tracking failed
```

**3. Two-phase kill:**
```bash
kill -TERM "$PID" && sleep 2 && kill -9 "$PID"
# Try graceful first, then force
```

---

## Common Modifications

### Modification 1: Add Email Notifications

**Add after line 325 (success banner):**
```bash
if command -v mail >/dev/null 2>&1; then
  echo "vLLM job ${SLURM_JOB_ID} started on ${SERVER}. Port forwarding: $(cat $PORT_FWD_FILE)" \
    | mail -s "vLLM Job Started" ${USER}@institution.edu
fi
```

### Modification 2: Auto-warmup Requests

**Add after line 273 (Gradio launch):**
```bash
# Send warmup requests to reduce first-user latency
for i in {1..5}; do
  curl -s "http://127.0.0.1:${VLLM_PORT}/v1/chat/completions" \
    -H "Content-Type: application/json" \
    -d '{"model":"'"${VLLM_MODEL}"'","messages":[{"role":"user","content":"hi"}],"max_tokens":1}' \
    > /dev/null 2>&1 &
done
```

### Modification 3: Persistent Conversation Storage

**Add before Gradio launch (line 279):**
```bash
export CONVERSATION_DIR="${WORK_DIR}/conversations/${SLURM_JOB_ID}"
mkdir -p "$CONVERSATION_DIR"
# Gradio app would need code changes to use this
```

### Modification 4: Custom Health Check Endpoint

**Add to monitoring loop (after line 356):**
```bash
# Check vLLM queue length
QUEUE_LEN=$(curl -s "http://127.0.0.1:${VLLM_PORT}/metrics" \
  | grep "vllm:num_requests_waiting" | awk '{print $2}')
if [ -n "$QUEUE_LEN" ] && [ "$QUEUE_LEN" -gt 10 ]; then
  echo "⚠️  High queue length: ${QUEUE_LEN} requests waiting"
fi
```

---

## Testing & Debugging

### Interactive Testing

```bash
# Run script interactively (not batch)
srun -p amd_a100nv_8 --gres=gpu:1 --pty bash
cd /scratch/$USER/gpt-oss-with-vllm-on-supercomputer
./vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B
# Watch output in real-time, Ctrl+C to stop
```

### Debugging Checklist

1. **Check SLURM allocation:**
   ```bash
   squeue -u $USER -o "%.18i %.9P %.8j %.8u %.2t %.10M %.6D %R %b"
   ```

2. **Verify SIF exists:**
   ```bash
   ls -lh /scratch/$USER/gpt-oss-with-vllm-on-supercomputer/vllm-gptoss.sif
   ```

3. **Check disk space:**
   ```bash
   df -h /scratch/$USER
   ```

4. **Test Singularity:**
   ```bash
   singularity exec --nv vllm-gptoss.sif python -c "import vllm; print(vllm.__version__)"
   ```

5. **Verify network access:**
   ```bash
   curl -I https://huggingface.co
   ```

---

## Summary

This SLURM script is a **production-grade orchestrator** that:
- Handles environment setup across diverse HPC clusters
- Provides detailed progress feedback for long operations
- Implements robust health monitoring and cleanup
- Generates user-friendly SSH tunnel commands
- Gracefully handles failures at multiple levels

**Key Strengths:**
- Defensive programming (check before use, fallbacks)
- User experience (emoji, progress bars, clear errors)
- Maintainability (well-structured, commented sections)

**Areas for Extension:**
- Email notifications
- Auto-scaling (launch more vLLM instances)
- Distributed tracing integration
- Automatic SSH tunnel establishment

---

**Document Version:** 1.0
**Last Updated:** 2025-01-27
