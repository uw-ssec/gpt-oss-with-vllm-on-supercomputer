# Development Guide

> Complete guide for setting up, testing, and contributing to this project

---

## Table of Contents

1. [Development Environment Setup](#development-environment-setup)
2. [Project Structure](#project-structure)
3. [Testing Strategy](#testing-strategy)
4. [Code Style & Conventions](#code-style--conventions)
5. [Making Changes](#making-changes)
6. [Testing Your Changes](#testing-your-changes)
7. [Contributing Workflow](#contributing-workflow)
8. [Common Development Tasks](#common-development-tasks)

---

## Development Environment Setup

### Prerequisites

**Required:**
- Access to SLURM-managed HPC cluster with NVIDIA GPUs
- SSH access to cluster
- Git installed locally and on cluster
- Basic familiarity with Bash, Python, SLURM

**Optional but Recommended:**
- Local Python 3.10+ for testing Gradio UI
- Docker/Singularity knowledge for container modifications
- Experience with vLLM or similar LLM inference engines

### Initial Setup on HPC Cluster

#### Step 1: Clone Repository

```bash
# Login to HPC cluster
ssh user@cluster.login.edu

# Navigate to scratch space (large models need space)
cd /scratch/$USER

# Clone repository
git clone https://github.com/hwang2006/gpt-oss-with-vllm-on-supercomputer.git
cd gpt-oss-with-vllm-on-supercomputer

# Verify structure
ls -la
# Should see: vllm_gradio_run_singularity.sh, vllm_web.py, README.md, etc.
```

#### Step 2: Build Singularity Container

```bash
# Build vLLM container (takes 15-30 minutes, ~10GB download)
singularity build --fakeroot vllm-gptoss.sif docker://vllm/vllm-openai:gptoss

# Verify build
singularity exec ./vllm-gptoss.sif python -c "import vllm; print(vllm.__version__)"
# Should print: 0.10.2 or similar

# Check vLLM CLI
singularity exec ./vllm-gptoss.sif vllm --help
```

**Troubleshooting:**
- `--fakeroot` requires Singularity 3.5+ and cluster support
- If fails, ask admin to build SIF centrally
- Alternative: Use pre-built SIF from shared location

#### Step 3: Setup Conda Environment (for Gradio)

```bash
# Load required modules (cluster-specific)
module load gcc/10.2.0 cuda/12.1

# Create conda environment
conda create -y -n vllm-hpc python=3.11
conda activate vllm-hpc

# Install Gradio
pip install gradio==5.42.0

# Verify installation
python -c "import gradio; print(gradio.__version__)"
# Should print: 5.42.0
```

**Alternative: Use requirements.txt**
```bash
pip install -r requirements.txt
```

#### Step 4: Configure Environment Variables

```bash
# Add to ~/.bashrc or ~/.bash_profile
export HF_HOME="/scratch/$USER/.huggingface"
export VLLM_CACHE_ROOT="/scratch/$USER/gpt-oss-with-vllm-on-supercomputer/.vllm"
export XDG_CACHE_HOME="/scratch/$USER/.gradio_cache"

# Create directories
mkdir -p "$HF_HOME/hub" "$VLLM_CACHE_ROOT" "$XDG_CACHE_HOME"

# Optional: Set HuggingFace token (for private models)
export HF_TOKEN="hf_your_token_here"
```

#### Step 5: Test Installation

```bash
# Interactive test (request 1 GPU for 1 hour)
srun -p <partition> --gres=gpu:1 --time=1:00:00 --pty bash

# Once on compute node
cd /scratch/$USER/gpt-oss-with-vllm-on-supercomputer
./vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B

# Watch for:
# - "🚀 Starting vLLM server..."
# - "✅ vLLM API is ready!"
# - "✅ Gradio UI is up!"
# - SSH port forwarding command printed

# In separate terminal on your laptop
ssh -L localhost:7860:gpu05:7860 -L localhost:8000:gpu05:8000 user@cluster.edu

# Open browser: http://localhost:7860
# Test chat: "Say hello"
```

---

## Project Structure

### Directory Layout

```
gpt-oss-with-vllm-on-supercomputer/
├── README.md                           # User-facing documentation
├── AGENTS.md                           # AI agent & developer guide (main entry)
├── LICENSE                             # MIT license
├── requirements.txt                    # Python dependencies (Gradio)
├── .gitignore                          # Version control exclusions
│
├── vllm_gradio_run_singularity.sh     # SLURM orchestration script (374 lines)
├── vllm_web.py                         # Gradio web interface (237 lines)
│
├── agents_docs/                        # Detailed technical documentation
│   ├── architecture.md                 # System architecture & design
│   ├── slurm_script_guide.md          # Deep dive into SLURM script
│   ├── gradio_app_guide.md            # Gradio implementation details
│   ├── development_guide.md           # This file
│   ├── deployment_guide.md            # Site-specific deployment
│   ├── api_reference.md               # vLLM API documentation
│   └── troubleshooting.md             # Common issues & solutions
│
├── assets/                             # Screenshots for documentation
│   ├── cmd_ui.png
│   └── gradio_ui_vllm.png
│
├── logs/                               # Runtime logs (created by script)
│   ├── vllm_server_<JOBID>.log
│   └── gradio_server_<JOBID>.log
│
├── .vllm/                              # vLLM compilation cache (created)
├── port_forwarding_<JOBID>.txt        # Generated SSH commands
└── vllm-gptoss.sif                    # Singularity container (~10GB)
```

### Key Files Explained

| File | Lines | Purpose | Modify When |
|------|-------|---------|-------------|
| `vllm_gradio_run_singularity.sh` | 374 | SLURM job orchestration | Adding features, tuning, monitoring |
| `vllm_web.py` | 237 | Web UI | UI changes, API updates |
| `README.md` | 377 | User docs | User-facing changes |
| `AGENTS.md` | ~400 | Dev guide entry point | Major architecture changes |
| `agents_docs/*.md` | ~3000 | Technical docs | Implementation details |

---

## Testing Strategy

### Testing Pyramid

```
        ┌──────────────────┐
        │  E2E Tests       │  Manual (full SLURM job)
        │  (Manual)        │
        └──────────────────┘
              ▲
        ┌────────────────────┐
        │ Integration Tests  │  Interactive srun
        │ (Semi-automated)   │
        └────────────────────┘
              ▲
        ┌──────────────────────────┐
        │   Unit Tests              │  Local Python/Bash
        │   (Automated where       │
        │    possible)              │
        └──────────────────────────┘
```

### Test Levels

#### Level 1: Unit Tests (Functions in Isolation)

**Bash Script Functions:**
```bash
# Test cleanup function (lines 99-110)
GRADIO_PID=12345
VLLM_PID=67890
cleanup
# Verify: processes killed, no errors printed

# Test argument parsing (lines 54-63)
bash vllm_gradio_run_singularity.sh --model test-model --vllm-port 9000 --help
# Verify: VLLM_MODEL="test-model", VLLM_PORT=9000
```

**Python Functions:**
```bash
# Test VLLMChat initialization
python3 -c "
from vllm_web import VLLMChat
chat = VLLMChat('http://localhost:8000/v1')
assert chat.base_url == 'http://localhost:8000/v1'
print('✓ VLLMChat init works')
"

# Test _strip_think function
python3 -c "
from vllm_web import _strip_think
text = '<think>reasoning</think>The answer is 42.'
result = _strip_think(text)
assert result == 'The answer is 42.'
print('✓ _strip_think works')
"
```

#### Level 2: Integration Tests (Components Together)

**Test vLLM API Directly:**
```bash
# Start vLLM in container (interactive)
singularity exec --nv vllm-gptoss.sif \
  vllm serve Qwen/Qwen3-0.6B --port 8000 &

# Wait for startup
sleep 30

# Test /v1/models endpoint
curl http://localhost:8000/v1/models | jq .

# Test chat completion
curl http://localhost:8000/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "Qwen/Qwen3-0.6B",
    "messages": [{"role": "user", "content": "Hi"}],
    "max_tokens": 10
  }' | jq .

# Cleanup
pkill -f "vllm serve"
```

**Test Gradio UI (Mock Backend):**
```bash
# Create mock vLLM server (see gradio_app_guide.md)
python mock_vllm.py &

# Start Gradio UI
python vllm_web.py --base-url http://localhost:8000/v1 &

# Test in browser or with automation
curl http://localhost:7860  # Should return HTML

# Cleanup
pkill -f "vllm_web.py"
pkill -f "mock_vllm.py"
```

#### Level 3: End-to-End Tests (Full System)

**Test via Interactive SLURM:**
```bash
# Request GPU
srun -p <partition> --gres=gpu:1 --time=1:00:00 --pty bash

# Run full script
cd /scratch/$USER/gpt-oss-with-vllm-on-supercomputer
./vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B

# Verify outputs:
# 1. vLLM starts without errors
# 2. Gradio UI launches
# 3. Port forwarding command generated
# 4. Can chat via UI
# 5. Ctrl+C cleanup works

# Check logs
tail -f logs/vllm_server_*.log
tail -f logs/gradio_server_*.log
```

**Test via Batch SLURM:**
```bash
# Submit job
sbatch vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B

# Monitor job
squeue -u $USER
watch -n 5 'tail -20 slurm-<JOBID>.out'

# Test SSH tunnel (from laptop)
ssh -L localhost:7860:gpu05:7860 -L localhost:8000:gpu05:8000 user@cluster.edu

# Open http://localhost:7860
# Send test messages

# Verify monitoring
# - Heartbeat logs every 5 minutes
# - GPU stats reported
# - API health checks pass

# Cleanup
scancel <JOBID>
```

### Test Checklist Before Committing

- [ ] Bash script has no syntax errors: `bash -n vllm_gradio_run_singularity.sh`
- [ ] Python script has no syntax errors: `python -m py_compile vllm_web.py`
- [ ] Help text is accurate: `./vllm_gradio_run_singularity.sh --help`
- [ ] Default model works: `sbatch vllm_gradio_run_singularity.sh`
- [ ] Custom model works: `sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b`
- [ ] Custom ports work: `sbatch vllm_gradio_run_singularity.sh --vllm-port 9000`
- [ ] Cleanup works: `scancel <JOBID>` (check no zombies: `ps aux | grep vllm`)
- [ ] Logs are readable: `cat logs/vllm_server_*.log`
- [ ] Port forwarding file generated: `cat port_forwarding_*.txt`
- [ ] UI displays correctly in browser
- [ ] Streaming responses work
- [ ] Model refresh button works
- [ ] Documentation updated if needed

---

## Code Style & Conventions

### Bash Script Style (vllm_gradio_run_singularity.sh)

**1. Variable Naming:**
```bash
# UPPERCASE for exported/global variables
VLLM_PORT=8000
export HF_HOME="/scratch/$USER/.huggingface"

# lowercase for local variables
local temp_file="/tmp/myfile"
```

**2. Function Definitions:**
```bash
# Functions without empty lines before closing brace
function_name() {
  echo "Do something"
  return 0
}
```

**3. Error Handling:**
```bash
# Use `set +e` for non-strict mode (we handle errors manually)
set +e

# Check command success explicitly
if ! command_that_might_fail; then
  echo "Error: command failed"
  exit 1
fi

# Use || true for commands that might fail harmlessly
pkill -f "vllm serve" 2>/dev/null || true
```

**4. Logging:**
```bash
# Prefix messages with emoji for visibility
echo "🚀 Starting service..."
echo "✅ Service ready!"
echo "⚠️  Warning: something"
echo "❌ Error: failed"
```

**5. Comments:**
```bash
# Section headers with visual separation
#######################################
# Section Name
#######################################

# Inline comments for complex logic
ELAPSED=$((ELAPSED+10))  # Increment by polling interval
```

### Python Style (vllm_web.py)

**1. Type Hints:**
```python
def chat_stream(self, messages: list, model: str, temperature: float, max_tokens: int) -> Generator[str, None, None]:
    """Streams content from vLLM API."""
    ...
```

**2. Docstrings:**
```python
def important_function():
    """
    Brief description of what the function does.

    For complex functions, include:
    - Args: parameter descriptions
    - Returns: return value description
    - Raises: exceptions that might be raised
    """
```

**3. Imports:**
```python
# Standard library first
import os
import time

# Third-party libraries
import requests
import gradio as gr

# Local modules (if any)
from .utils import helper_function
```

**4. Constants:**
```python
# UPPERCASE for module-level constants
DEFAULT_BASE_URL = os.getenv("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
FILTER_THINK = True
```

**5. Error Handling:**
```python
try:
    # Specific operation
    result = risky_operation()
except SpecificException as e:
    # Handle specific error
    print(f"[context] Error: {e}")
    # Graceful degradation
    return default_value
except Exception as e:
    # Catch-all for unexpected errors
    print(f"[unexpected] Error: {e}")
    raise  # Re-raise if can't handle
```

### Documentation Style

**1. Markdown Files:**
- Use ATX headers (`#`, `##`, `###`)
- Include table of contents for long docs
- Use code fences with language tags
- Include file paths with absolute references

**2. Code Comments:**
- Explain *why*, not *what* (code should be self-explanatory)
- Add TODO comments for future work: `# TODO: Add retry logic`
- Reference line numbers in docs: `# See lines 162-187`

**3. Commit Messages:**
```
<type>: <short summary> (50 chars max)

<detailed description if needed>

- Bullet points for multiple changes
- Reference issues: Fixes #123
```

Types: `feat`, `fix`, `docs`, `style`, `refactor`, `test`, `chore`

---

## Making Changes

### Workflow for New Features

**1. Plan the Change**
- Identify affected files
- Consider backward compatibility
- Check if documentation needs updates
- Estimate testing effort

**2. Create Development Branch (if applicable)**
```bash
git checkout -b feature/add-conversation-export
```

**3. Implement the Change**
- Follow code style conventions
- Add comments for complex logic
- Keep commits focused (one logical change per commit)

**4. Test the Change**
- Run unit tests
- Test interactively with `srun`
- Test in batch mode with `sbatch`
- Verify logs and outputs

**5. Update Documentation**
- Update AGENTS.md if architecture changed
- Update README.md if user-facing
- Add to agents_docs/ if implementation detail
- Update code comments

**6. Commit and Push**
```bash
git add <files>
git commit -m "feat: add conversation export feature

- Add export button to Gradio UI
- Save conversations as JSON to disk
- Update documentation with usage example"

git push origin feature/add-conversation-export
```

**7. Create Pull Request (if applicable)**
- Describe the change clearly
- Include testing steps
- Reference related issues

### Common Change Patterns

#### Pattern 1: Add vLLM CLI Flag

**File:** `vllm_gradio_run_singularity.sh` (lines 190-204)

```bash
# Add new flag to vllm serve command
nohup singularity exec --nv "$SIF_PATH" \
  vllm serve "${VLLM_MODEL}" \
    --host 0.0.0.0 \
    --port "${VLLM_PORT}" \
    # ... existing flags ...
    --enable-prefix-caching \  # NEW FLAG
    > "$VLLM_LOG" 2>&1 &
```

**Testing:**
1. Check vLLM logs for new flag
2. Verify performance impact
3. Update README with new capability

#### Pattern 2: Add Gradio UI Component

**File:** `vllm_web.py` (lines 129-160)

```python
# Add component to layout
with gr.Column(scale=3, elem_id="side_col"):
    # ... existing components ...

    # NEW: System prompt textbox
    system_prompt = gr.Textbox(
        label="System Prompt",
        value="You are a helpful assistant.",
        lines=3
    )
```

**Update event handler:**
```python
def on_send(msg, hist, server, model, temp, max_toks, sys_prompt):  # Add parameter
    # ... existing code ...
    messages = [{"role": "system", "content": sys_prompt}] + new_hist
    # ... rest of function ...

# Wire the input
send.click(on_send, inputs=[..., system_prompt], outputs=[...])
```

**Testing:**
1. Verify UI renders correctly
2. Test with different system prompts
3. Check message history structure

#### Pattern 3: Add Environment Variable

**File:** `vllm_gradio_run_singularity.sh` (lines 21-40)

```bash
# Add new variable with default
ENABLE_METRICS="${ENABLE_METRICS:-false}"

# Use in vLLM command
if [ "$ENABLE_METRICS" = "true" ]; then
  METRICS_FLAG="--enable-metrics"
else
  METRICS_FLAG=""
fi

nohup singularity exec --nv "$SIF_PATH" \
  vllm serve "${VLLM_MODEL}" \
    # ... other flags ...
    $METRICS_FLAG \
    > "$VLLM_LOG" 2>&1 &
```

**Usage:**
```bash
export ENABLE_METRICS=true
sbatch vllm_gradio_run_singularity.sh
```

**Testing:**
1. Test with default (false)
2. Test with enabled (true)
3. Check vLLM logs for metrics output

---

## Testing Your Changes

### Pre-Commit Checks

```bash
# 1. Syntax check Bash script
bash -n vllm_gradio_run_singularity.sh
echo $?  # Should be 0

# 2. Syntax check Python script
python -m py_compile vllm_web.py
echo $?  # Should be 0

# 3. Check for common issues
# - No trailing whitespace
# - No tabs (use spaces)
# - Unix line endings (LF not CRLF)
dos2unix vllm_gradio_run_singularity.sh vllm_web.py  # If needed

# 4. Lint Python (optional but recommended)
pip install ruff
ruff check vllm_web.py
ruff format vllm_web.py  # Auto-format

# 5. Check documentation links
# Manually verify all internal links in AGENTS.md and agents_docs/
```

### Interactive Testing Workflow

```bash
# 1. Request interactive session
srun -p <partition> --gres=gpu:1 --time=1:00:00 --pty bash

# 2. Navigate to repo
cd /scratch/$USER/gpt-oss-with-vllm-on-supercomputer

# 3. Activate conda env
conda activate vllm-hpc

# 4. Run script with test model
./vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B

# 5. Watch logs in another terminal
ssh user@cluster.edu
tail -f /scratch/$USER/gpt-oss-with-vllm-on-supercomputer/logs/vllm_server_*.log

# 6. Test UI from laptop
# Copy SSH command from port_forwarding_*.txt
ssh -L localhost:7860:gpu05:7860 -L localhost:8000:gpu05:8000 user@cluster.edu

# Open http://localhost:7860
# Test chat functionality

# 7. Verify monitoring
# Wait for heartbeat logs (every 5 minutes)
# Check GPU stats are displayed

# 8. Test cleanup
# Ctrl+C in srun session
# Verify processes killed: ps aux | grep vllm
```

### Batch Testing Workflow

```bash
# 1. Submit batch job
JOBID=$(sbatch vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B | awk '{print $4}')
echo "Job ID: $JOBID"

# 2. Monitor job status
watch -n 5 "squeue -j $JOBID -o '%.10i %.9P %.30j %.8u %.2t %.10M %.6D %R'"

# 3. Watch SLURM output
tail -f slurm-${JOBID}.out

# 4. Once started, test SSH tunnel
# Get node name from slurm output (e.g., "Server: gpu05")
NODE=$(grep "Server:" slurm-${JOBID}.out | awk '{print $2}')
ssh -L localhost:7860:${NODE}:7860 -L localhost:8000:${NODE}:8000 user@cluster.edu

# 5. Test functionality
# Open http://localhost:7860
# Send test messages
# Verify responses

# 6. Check logs
ssh user@cluster.edu
tail -100 /scratch/$USER/gpt-oss-with-vllm-on-supercomputer/logs/vllm_server_${JOBID}.log
tail -100 /scratch/$USER/gpt-oss-with-vllm-on-supercomputer/logs/gradio_server_${JOBID}.log

# 7. Cleanup
scancel $JOBID

# 8. Verify cleanup
sleep 10
ssh user@cluster.edu "ps aux | grep vllm"  # Should show no processes
```

---

## Contributing Workflow

### For Repository Maintainers

**1. Direct Commits to Main**
```bash
git pull origin main
# Make changes
git add <files>
git commit -m "feat: description"
git push origin main
```

**2. Branch-Based Development**
```bash
git checkout -b feature/new-feature
# Make changes
git add <files>
git commit -m "feat: description"
git push origin feature/new-feature
# Create PR on GitHub
```

### For External Contributors

**1. Fork Repository**
- Click "Fork" on GitHub
- Clone your fork: `git clone https://github.com/yourusername/gpt-oss-with-vllm-on-supercomputer.git`

**2. Create Feature Branch**
```bash
git checkout -b feature/your-feature
```

**3. Make Changes**
- Follow code style conventions
- Add tests
- Update documentation

**4. Commit Changes**
```bash
git add <files>
git commit -m "feat: description"
```

**5. Push to Fork**
```bash
git push origin feature/your-feature
```

**6. Create Pull Request**
- Go to original repository on GitHub
- Click "New Pull Request"
- Select your fork and branch
- Describe changes clearly

**7. Address Review Comments**
```bash
# Make requested changes
git add <files>
git commit -m "fix: address review comments"
git push origin feature/your-feature
# PR updates automatically
```

---

## Common Development Tasks

### Task 1: Add Support for New vLLM Version

**Steps:**
1. Update Singularity build command:
   ```bash
   singularity build --fakeroot vllm-gptoss-new.sif docker://vllm/vllm-openai:v0.11.0
   ```

2. Test with existing script:
   ```bash
   export SIF_PATH="$PWD/vllm-gptoss-new.sif"
   ./vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B
   ```

3. Update documentation:
   - README.md badge
   - Build instructions
   - Compatibility notes

### Task 2: Add New Model to Documentation

**Steps:**
1. Test model:
   ```bash
   sbatch vllm_gradio_run_singularity.sh --model <new-model-id>
   ```

2. Document requirements:
   - GPU memory needed
   - Context window
   - Special flags

3. Update README.md:
   - Add to examples
   - Update model compatibility table

### Task 3: Debug Performance Issue

**Steps:**
1. Enable verbose logging:
   ```bash
   # Add to vLLM command
   --log-level debug
   ```

2. Profile GPU usage:
   ```bash
   # In monitoring loop, add:
   nvidia-smi dmon -s pucvmet -c 60
   ```

3. Analyze logs:
   ```bash
   grep -i "latency\|throughput\|memory" logs/vllm_server_*.log
   ```

4. Experiment with tuning:
   ```bash
   export MAX_NUM_SEQS=32  # Reduce batch size
   export GPU_MEM_UTIL=0.85  # More headroom
   ```

### Task 4: Add Automated Tests

**Create `tests/` directory:**
```bash
mkdir -p tests

# tests/test_vllm_web.py
import sys
sys.path.insert(0, '..')
from vllm_web import VLLMChat, _strip_think

def test_strip_think():
    text = "<think>reasoning</think>Answer"
    assert _strip_think(text) == "Answer"

def test_vllm_chat_init():
    chat = VLLMChat("http://localhost:8000/v1")
    assert chat.base_url == "http://localhost:8000/v1"

if __name__ == "__main__":
    test_strip_think()
    test_vllm_chat_init()
    print("All tests passed!")
```

**Run tests:**
```bash
python tests/test_vllm_web.py
```

---

## Summary

This development guide provides:
- Complete setup instructions for HPC environment
- Testing strategies at multiple levels
- Code style conventions for consistency
- Change patterns for common modifications
- Contributing workflow for collaboration

**Key Principles:**
- Test changes interactively before batch submission
- Follow existing code style
- Update documentation with code changes
- Commit focused, logical changes
- Clean up test jobs and resources

**Next Steps:**
- Set up development environment
- Run through test workflow
- Make a small change to familiarize yourself
- Read detailed guides in `agents_docs/`

---

**Document Version:** 1.0
**Last Updated:** 2025-01-27
