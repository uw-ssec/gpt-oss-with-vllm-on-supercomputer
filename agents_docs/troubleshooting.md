# Troubleshooting Guide

> Comprehensive solutions for common issues and debugging strategies

---

## Table of Contents

1. [Quick Diagnostics](#quick-diagnostics)
2. [Startup Issues](#startup-issues)
3. [Model Loading Problems](#model-loading-problems)
4. [Performance Issues](#performance-issues)
5. [Network & Connectivity](#network--connectivity)
6. [GPU & Memory Issues](#gpu--memory-issues)
7. [UI & Interface Problems](#ui--interface-problems)
8. [SLURM Job Issues](#slurm-job-issues)
9. [Debugging Strategies](#debugging-strategies)
10. [Getting Help](#getting-help)

---

## Quick Diagnostics

### Health Check Checklist

Run these commands to quickly assess system status:

```bash
# 1. Check SLURM job status
squeue -u $USER
# Look for: Running (R) or Pending (PD) jobs

# 2. Check GPU availability
ssh <compute-node>
nvidia-smi
# Look for: GPU processes, memory usage

# 3. Check vLLM server
curl http://localhost:8000/v1/models
# Expected: JSON with model list

# 4. Check Gradio UI
curl -I http://localhost:7860
# Expected: HTTP/1.1 200 OK

# 5. Check logs for errors
tail -50 logs/vllm_server_*.log | grep -i error
tail -50 logs/gradio_server_*.log | grep -i error
```

### Common Quick Fixes

| Symptom | Quick Fix |
|---------|-----------|
| Port already in use | Use different ports: `--vllm-port 9000 --gradio-port 7000` |
| GPU out of memory | Use smaller model or reduce `--max-model-len` |
| vLLM not responding | Check logs, wait longer (first run takes time) |
| SSH tunnel broken | Re-run SSH command from `port_forwarding_<JOBID>.txt` |
| Gradio shows "No models" | Click "Refresh Models" button |

---

## Startup Issues

### Issue 1: "vLLM API not responding" (Stuck at waiting)

**Symptoms:**
```
⏳ Still preparing vLLM API… (120s)
⏳ Still preparing vLLM API… (130s)
...
```

**Causes:**
1. First-time model download (can take 10-30 minutes)
2. CUDA graph compilation (2-5 minutes)
3. Out of memory (OOM) during initialization
4. Network issue preventing model download
5. vLLM crashed silently

**Diagnostic Steps:**
```bash
# Check vLLM log for progress
tail -f logs/vllm_server_<JOBID>.log

# Look for:
# - "Downloading..." (model download in progress)
# - "Loading weights took..." (weights loaded successfully)
# - "Graph capturing..." (CUDA compilation)
# - "Application startup complete" (ready!)
# - "OutOfMemoryError" (OOM issue)
```

**Solutions:**

**A. First download is slow (normal):**
```bash
# Wait patiently, check progress in logs
# Model downloads only once, cached for future runs
# Progress bars enabled with --use-tqdm-on-load
```

**B. Out of memory:**
```bash
# Try smaller model
sbatch vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B

# Or reduce context window
export MAX_MODEL_LEN=2048  # Down from 40960
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b

# Or use tensor parallelism (2+ GPUs)
#SBATCH --gres=gpu:2
export TP_SIZE=2
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b
```

**C. Network issue:**
```bash
# Check internet access from compute node
ssh <compute-node>
curl -I https://huggingface.co
# If fails, compute nodes may not have internet access

# Pre-download model on login node (if has internet)
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/gpt-oss-20b', cache_dir='/scratch/$USER/.huggingface/hub')
"

# Then submit job (uses cached model)
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b
```

**D. vLLM crashed:**
```bash
# Check for crash in logs
grep -i "error\|exception\|traceback" logs/vllm_server_<JOBID>.log

# Common crashes:
# - CUDA driver version mismatch
# - Unsupported model architecture
# - Insufficient GPU memory

# Try different model
sbatch vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B
```

---

### Issue 2: "Gradio process exited" immediately

**Symptoms:**
```
🌐 Starting Gradio web interface...
Gradio PID: 123456
❌ Gradio process exited. Last logs:
Traceback (most recent call last):
  ...
```

**Causes:**
1. Python version too old (< 3.8)
2. Gradio not installed
3. vLLM API not running (Gradio waits for it)
4. Port already in use

**Diagnostic Steps:**
```bash
# Check Python version
python --version
# Need: Python 3.8+

# Check Gradio installation
python -c "import gradio; print(gradio.__version__)"
# Expected: 5.42.0

# Check port availability
ss -tuln | grep 7860
# If shows LISTEN, port is taken
```

**Solutions:**

**A. Python too old:**
```bash
# Create new conda environment
conda create -n vllm-hpc python=3.11
conda activate vllm-hpc
pip install gradio==5.42.0

# Verify
python --version  # Should show Python 3.11.x
```

**B. Gradio not installed:**
```bash
pip install gradio==5.42.0
```

**C. Port conflict:**
```bash
# Use different port
sbatch vllm_gradio_run_singularity.sh --gradio-port 7100
```

---

### Issue 3: "Singularity: command not found"

**Symptoms:**
```
./vllm_gradio_run_singularity.sh: line 190: singularity: command not found
```

**Causes:**
- Singularity not installed on cluster
- Singularity not in PATH
- Module not loaded

**Solutions:**

**A. Load Singularity module:**
```bash
# Check available modules
module avail singularity

# Load module
module load singularity/3.8.0  # Version may vary

# Add to script (after line 133)
module load singularity/3.8.0
```

**B. Use Apptainer (renamed Singularity):**
```bash
# Some clusters renamed Singularity to Apptainer
module load apptainer

# Singularity commands still work (symlinked)
singularity --version
```

**C. Ask admin to install:**
```bash
# If not available, contact HPC support
# Alternative: Use Docker (if available and allowed)
```

---

## Model Loading Problems

### Issue 4: "Model download fails" or "Connection timeout"

**Symptoms:**
```
Error: Failed to download model
requests.exceptions.ConnectionError: HTTPSConnectionPool...
```

**Causes:**
1. No internet access from compute nodes
2. Firewall blocking HuggingFace
3. HuggingFace API rate limit
4. Model requires authentication (private model)

**Solutions:**

**A. Pre-download on login node:**
```bash
# Login node usually has internet
cd /scratch/$USER
export HF_HOME="/scratch/$USER/.huggingface"

# Download model
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/gpt-oss-20b', cache_dir='$HF_HOME/hub')
"

# Now submit job (uses cached model)
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b
```

**B. Use HuggingFace token (for private models):**
```bash
# Get token from https://huggingface.co/settings/tokens
export HF_TOKEN="hf_your_token_here"

# Add to job script (line 40)
export HF_TOKEN="${HF_TOKEN}"

sbatch vllm_gradio_run_singularity.sh --model <private-model>
```

**C. Use proxy (if cluster requires):**
```bash
# Add to script (after line 40)
export HTTP_PROXY="http://proxy.cluster.edu:8080"
export HTTPS_PROXY="http://proxy.cluster.edu:8080"
```

---

### Issue 5: "Unsupported model architecture"

**Symptoms:**
```
ValueError: Model architecture 'CustomLLamaForCausalLM' is not supported
```

**Causes:**
- Model uses custom architecture not in vLLM
- Model too new (vLLM version outdated)
- Model requires special flags

**Solutions:**

**A. Use `--trust-remote-code` (already enabled):**
```bash
# Script already includes this flag (line 201)
# Allows custom model code execution
```

**B. Update vLLM container:**
```bash
# Build newer vLLM version
singularity build --fakeroot vllm-latest.sif docker://vllm/vllm-openai:latest

# Use new container
export SIF_PATH="$PWD/vllm-latest.sif"
sbatch vllm_gradio_run_singularity.sh --model <model>
```

**C. Check vLLM supported models:**
```bash
# List supported architectures
singularity exec vllm-gptoss.sif python -c "
from vllm.model_executor.models import ModelRegistry
print('Supported models:')
for model in ModelRegistry.get_supported_archs():
    print(f'  - {model}')
"
```

---

## Performance Issues

### Issue 6: "Very slow inference" (< 5 tokens/sec)

**Symptoms:**
- First token latency > 10 seconds
- Tokens per second < 5 (expected: 20-30 for 20B model)

**Causes:**
1. GPU not used (CPU inference)
2. Memory swapping
3. Over-committed GPU (too many concurrent requests)
4. Model too large for GPU

**Diagnostic Steps:**
```bash
# Check GPU utilization
nvidia-smi dmon -s u
# Expected: GPU Utilization > 80%

# Check if CUDA available
singularity exec --nv vllm-gptoss.sif python -c "
import torch
print(f'CUDA available: {torch.cuda.is_available()}')
print(f'CUDA devices: {torch.cuda.device_count()}')
"
```

**Solutions:**

**A. Ensure GPU acceleration:**
```bash
# Verify --nv flag in script (line 190)
# This passes NVIDIA GPUs to container

# Check CUDA_VISIBLE_DEVICES
export CUDA_VISIBLE_DEVICES=0  # Use GPU 0
```

**B. Reduce batch size:**
```bash
export MAX_NUM_SEQS=16  # Down from 64
sbatch vllm_gradio_run_singularity.sh
```

**C. Reduce context window:**
```bash
export MAX_MODEL_LEN=2048  # Down from 40960
sbatch vllm_gradio_run_singularity.sh
```

**D. Use smaller model:**
```bash
sbatch vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B
```

---

### Issue 7: "High GPU memory usage" (GPU OOM)

**Symptoms:**
```
torch.cuda.OutOfMemoryError: CUDA out of memory
```

**Causes:**
- Model too large
- Context window too long
- Too many concurrent requests
- Memory leak (rare)

**Solutions:**

**A. Reduce GPU memory utilization:**
```bash
export GPU_MEM_UTIL=0.75  # Down from 0.90
sbatch vllm_gradio_run_singularity.sh
```

**B. Reduce context window:**
```bash
export MAX_MODEL_LEN=2048
sbatch vllm_gradio_run_singularity.sh
```

**C. Use tensor parallelism (multi-GPU):**
```bash
# Modify SLURM directives (line 16)
#SBATCH --gres=gpu:2

# Set tensor parallel size
export TP_SIZE=2
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b
```

**D. Use quantization:**
```bash
# Add to vllm serve command (line 190-204)
--quantization awq  # Or bitsandbytes, fp8
```

---

## Network & Connectivity

### Issue 8: "SSH tunnel not working" (Can't access UI)

**Symptoms:**
- Browser shows "Connection refused" at localhost:7860
- SSH command runs but ports not accessible

**Causes:**
1. Wrong node name in SSH command
2. Services not started yet
3. Firewall blocking ports
4. SSH tunnel died

**Diagnostic Steps:**
```bash
# Verify services running on compute node
ssh <compute-node>
ss -tuln | grep -E '7860|8000'
# Should show LISTEN on both ports

# Test local access on compute node
curl http://localhost:7860
curl http://localhost:8000/v1/models
```

**Solutions:**

**A. Verify node name:**
```bash
# Get node name from SLURM output
cat slurm-<JOBID>.out | grep "Server:"
# Example: Server: gpu05

# Use correct node in SSH command
ssh -L localhost:7860:gpu05:7860 -L localhost:8000:gpu05:8000 user@cluster.edu
```

**B. Check SSH tunnel is active:**
```bash
# On laptop, check SSH process
ps aux | grep "ssh -L"

# Re-establish if needed
ssh -L localhost:7860:gpu05:7860 -L localhost:8000:gpu05:8000 user@cluster.edu
```

**C. Try different local ports:**
```bash
# If localhost:7860 is taken
ssh -L localhost:7900:gpu05:7860 -L localhost:8100:gpu05:8000 user@cluster.edu

# Open http://localhost:7900
```

**D. Use ProxyJump:**
```bash
# If cluster has gateway node
ssh -J gateway@cluster.edu -L 7860:localhost:7860 -L 8000:localhost:8000 user@gpu05
```

---

### Issue 9: "Port already in use"

**Symptoms:**
```
OSError: [Errno 98] Address already in use
```

**Causes:**
- Previous job didn't cleanup
- Another user on same node
- System service using port

**Solutions:**

**A. Use different ports:**
```bash
sbatch vllm_gradio_run_singularity.sh --vllm-port 9000 --gradio-port 7100
```

**B. Kill stale processes:**
```bash
# Find processes using ports
lsof -i :8000
lsof -i :7860

# Kill them
pkill -f "vllm serve"
pkill -f "vllm_web.py"
```

**C. Wait for previous job to cleanup:**
```bash
# Check if previous job still running
squeue -u $USER

# If not, wait 30 seconds for cleanup
sleep 30
```

---

## GPU & Memory Issues

### Issue 10: "No GPU detected" in container

**Symptoms:**
```
RuntimeError: No CUDA GPUs are available
```

**Causes:**
- `--nv` flag missing
- CUDA driver mismatch
- GPU not allocated by SLURM
- Wrong CUDA_VISIBLE_DEVICES

**Solutions:**

**A. Verify SLURM GPU allocation:**
```bash
# Check job allocation
scontrol show job <JOBID> | grep "TRES"
# Should show: gres/gpu=1 (or more)

# Check GPU visibility
echo $CUDA_VISIBLE_DEVICES
# Should show: 0 (or 0,1,2...)
```

**B. Test GPU in container:**
```bash
singularity exec --nv vllm-gptoss.sif nvidia-smi
# Should show GPU info
```

**C. Check CUDA compatibility:**
```bash
# Host CUDA version
nvidia-smi | grep "CUDA Version"

# Container CUDA version
singularity exec --nv vllm-gptoss.sif nvcc --version

# Versions should be compatible (driver >= runtime)
```

---

### Issue 11: "GPU memory leak" (memory grows over time)

**Symptoms:**
- GPU memory usage increases with each request
- Eventually hits OOM
- Restart fixes temporarily

**Diagnostic Steps:**
```bash
# Monitor memory over time
watch -n 5 nvidia-smi
# Note: Memory used increases even when idle

# Check for memory fragmentation
nvidia-smi --query-gpu=memory.used,memory.free --format=csv -l 10
```

**Solutions:**

**A. Restart vLLM periodically:**
```bash
# Add to monitoring loop (after line 345)
if (( NOW - START > 43200 )); then  # 12 hours
  echo "⚠️  Restarting vLLM to clear memory"
  kill -TERM "$VLLM_PID"
  sleep 5
  # Re-launch vLLM (copy lines 190-207)
fi
```

**B. Reduce batch size:**
```bash
export MAX_NUM_SEQS=16  # Down from 64
```

**C. Update vLLM:**
```bash
# Build latest version (may have memory leak fixes)
singularity build --fakeroot vllm-latest.sif docker://vllm/vllm-openai:latest
```

---

## UI & Interface Problems

### Issue 12: Gradio shows "No models available"

**Symptoms:**
- Dropdown shows "No models available"
- Can't send messages

**Causes:**
1. vLLM not fully started yet
2. vLLM crashed
3. Network issue between Gradio and vLLM
4. Wrong base URL

**Solutions:**

**A. Wait and refresh:**
```bash
# Click "🔄 Refresh Models" button in UI
# Wait 10 seconds, try again
```

**B. Check vLLM status:**
```bash
curl http://localhost:8000/v1/models
# If returns JSON: vLLM is up
# If fails: vLLM is down
```

**C. Verify Gradio configuration:**
```python
# Check in vllm_web.py (line 12)
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
# Should match vLLM port
```

**D. Restart Gradio:**
```bash
# Kill Gradio
pkill -f "vllm_web.py"

# Restart (script auto-restarts on failure)
# Or manually:
export OPENAI_BASE_URL="http://127.0.0.1:8000/v1"
python vllm_web.py --host 0.0.0.0 --port 7860 &
```

---

### Issue 13: "Streaming responses don't work" (text appears all at once)

**Symptoms:**
- Entire response appears instantly (not word-by-word)
- Loading indicator shows, then full text

**Causes:**
- Buffering in HTTP stack
- Gradio version incompatibility
- Browser issue

**Solutions:**

**A. Update Gradio:**
```bash
pip install --upgrade gradio
# Should be >= 5.0
```

**B. Check SSE parsing:**
```python
# In vllm_web.py, check chat_stream function (line 51-85)
# Ensure: stream=True in request
# Ensure: yielding chunks (not accumulating)
```

**C. Test with curl:**
```bash
# Verify vLLM streams correctly
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"openai/gpt-oss-20b","messages":[{"role":"user","content":"Count to 10"}],"max_tokens":50,"stream":true}'

# Should see: data: {...} lines appear progressively
```

---

### Issue 14: "Chat history lost on refresh"

**Symptoms:**
- Browser refresh clears conversation
- No way to restore history

**Cause:**
- Conversations stored in browser memory (not persistent)

**Solution:**

**A. Expected behavior:**
This is by design (stateless). Conversations not saved to disk.

**B. Add persistence (customization):**
```python
# Modify vllm_web.py to save to file
import json
from pathlib import Path

HISTORY_DIR = Path("/scratch/$USER/conversations")
HISTORY_DIR.mkdir(exist_ok=True)

def save_conversation(hist, session_id):
    file_path = HISTORY_DIR / f"{session_id}.json"
    with open(file_path, "w") as f:
        json.dump(hist, f)

def load_conversation(session_id):
    file_path = HISTORY_DIR / f"{session_id}.json"
    if file_path.exists():
        with open(file_path) as f:
            return json.load(f)
    return []
```

---

## SLURM Job Issues

### Issue 15: "Job stays in pending (PD) state"

**Symptoms:**
```
$ squeue -u $USER
JOBID  PARTITION  NAME       USER  ST  TIME  NODES  NODELIST(REASON)
12345  gpu        vllm_grad  user  PD  0:00  1      (Resources)
```

**Causes:**
1. No available GPUs (all busy)
2. Resource request exceeds available (e.g., asking for 8 GPUs on 4-GPU nodes)
3. Job priority low
4. Partition down/maintenance

**Solutions:**

**A. Check queue:**
```bash
squeue -p <partition>
# See how many jobs ahead of yours
```

**B. Check resource availability:**
```bash
sinfo -p <partition> -o "%20P %5D %14F %8z %10m %10G"
# Look at: NODES(A/I/O/T) and GRES (gpu:N)
```

**C. Reduce resource request:**
```bash
# Try different partition with more availability
#SBATCH --partition=<other-partition>

# Or reduce GPU count
#SBATCH --gres=gpu:1  # Instead of gpu:2
```

**D. Check job priority:**
```bash
sprio -j <JOBID>
# Higher priority = runs sooner
```

---

### Issue 16: "Job killed immediately after start"

**Symptoms:**
```
$ squeue -u $USER
(no jobs)

$ sacct -j <JOBID>
JOBID       State      ExitCode
12345       FAILED     1:0
```

**Causes:**
1. Syntax error in script
2. Module load failure
3. SIF file not found
4. Permission denied

**Solutions:**

**A. Check SLURM output:**
```bash
cat slurm-<JOBID>.out
# Look for error messages
```

**B. Test script interactively:**
```bash
# Request interactive session
srun -p <partition> --gres=gpu:1 --pty bash

# Run script manually
cd /scratch/$USER/gpt-oss-with-vllm-on-supercomputer
./vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B
# Watch for errors
```

**C. Check file permissions:**
```bash
ls -la vllm_gradio_run_singularity.sh
# Should be: -rwxr-xr-x (executable)

chmod +x vllm_gradio_run_singularity.sh  # If not
```

---

## Debugging Strategies

### Strategy 1: Enable Verbose Logging

**vLLM:**
```bash
# Add to vllm serve command (line 193)
--log-level debug
```

**Python:**
```python
# Add to vllm_web.py (top of file)
import logging
logging.basicConfig(level=logging.DEBUG)
```

**Bash:**
```bash
# Add to script (line 2)
set -x  # Print each command before execution
```

### Strategy 2: Isolate Components

**Test vLLM alone:**
```bash
singularity exec --nv vllm-gptoss.sif \
  vllm serve Qwen/Qwen3-0.6B --port 8000 &
curl http://localhost:8000/v1/models
```

**Test Gradio alone (with mock backend):**
```bash
# Create mock server (see gradio_app_guide.md)
python mock_vllm.py &
python vllm_web.py --base-url http://localhost:8000/v1
```

### Strategy 3: Binary Search for Issues

**If script works on some nodes but not others:**
```bash
# Request specific node
srun -w gpu05 ./vllm_gradio_run_singularity.sh ...
# Test on working node, then failing node
# Compare: nvidia-smi, module list, disk space
```

### Strategy 4: Compare with Working Example

**Use smallest model for testing:**
```bash
sbatch vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B
# If works: issue is model-specific or resource-related
# If fails: issue is environment or configuration
```

---

## Getting Help

### Information to Provide

When seeking help, include:

**1. Environment:**
```bash
echo "Cluster: $(hostname)"
echo "SLURM version: $(sinfo --version)"
echo "Singularity version: $(singularity --version)"
echo "CUDA version: $(nvidia-smi | grep 'CUDA Version')"
echo "Python version: $(python --version)"
```

**2. Job details:**
```bash
echo "Job ID: $SLURM_JOB_ID"
echo "Partition: $SLURM_JOB_PARTITION"
echo "GPUs: $(echo $CUDA_VISIBLE_DEVICES)"
```

**3. Relevant logs:**
```bash
# Last 100 lines of vLLM log
tail -100 logs/vllm_server_<JOBID>.log

# Last 100 lines of Gradio log
tail -100 logs/gradio_server_<JOBID>.log

# SLURM output
tail -100 slurm-<JOBID>.out
```

**4. Error messages:**
Copy full error messages (not paraphrases).

**5. What you've tried:**
List troubleshooting steps already attempted.

### Where to Get Help

1. **GitHub Issues:** https://github.com/hwang2006/gpt-oss-with-vllm-on-supercomputer/issues
2. **HPC Support:** Contact cluster administrators for cluster-specific issues
3. **vLLM Community:** https://github.com/vllm-project/vllm/issues
4. **Gradio Community:** https://github.com/gradio-app/gradio/issues

---

## Summary

This troubleshooting guide covers:
- **Startup issues:** vLLM/Gradio not starting
- **Model loading:** Download failures, unsupported architectures
- **Performance:** Slow inference, memory issues
- **Network:** SSH tunneling, port conflicts
- **GPU:** Memory leaks, OOM errors
- **UI:** Model dropdown, streaming, persistence
- **SLURM:** Job pending, crashes
- **Debugging:** Systematic approaches to isolate issues

**General Debugging Workflow:**
1. Check quick diagnostics (health checks)
2. Review logs for error messages
3. Isolate the failing component
4. Test with minimal example
5. Compare with working configuration
6. Seek help with detailed information

**Prevention Best Practices:**
- Test with small models first
- Use interactive sessions for development
- Monitor GPU memory and logs
- Keep vLLM/Gradio updated
- Document working configurations

---

**Document Version:** 1.0
**Last Updated:** 2025-01-27
