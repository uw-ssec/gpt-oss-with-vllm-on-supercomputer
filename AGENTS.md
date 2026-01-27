# AI Agent & Developer Guide: GPT-OSS with vLLM on Supercomputer

> **Primary Entry Point for AI Coding Agents and Human Developers**

This document serves as the comprehensive guide for understanding, working with, and contributing to this codebase. It is designed to help both AI agents and human developers quickly understand the project's purpose, architecture, and development workflow.

---

## Table of Contents

- [Quick Start](#quick-start)
- [Project Overview](#project-overview)
- [Codebase Architecture](#codebase-architecture)
- [Key Files Reference](#key-files-reference)
- [Development Workflow](#development-workflow)
- [Detailed Documentation](#detailed-documentation)
- [Common Tasks](#common-tasks)
- [Troubleshooting Guide](#troubleshooting-guide)

---

## Quick Start

### For AI Agents
```bash
# Repository root
cd /Users/lsetiawan/Repos/SSEC/gpt-oss-with-vllm-on-supercomputer

# Key files to understand first
# 1. vllm_gradio_run_singularity.sh - SLURM job orchestration
# 2. vllm_web.py - Gradio web interface
# 3. README.md - User-facing documentation
# 4. agents_docs/ - Detailed technical documentation
```

### For Human Developers
1. Read [README.md](README.md) for user-facing documentation
2. Review this guide for development context
3. Explore [agents_docs/](agents_docs/) for detailed technical documentation
4. Check [agents_docs/development_guide.md](agents_docs/development_guide.md) for contribution guidelines

---

## Project Overview

### Purpose
This project enables running **GPT-OSS** (OpenAI's open-source GPT models) and other Hugging Face models on HPC (High-Performance Computing) clusters using:
- **vLLM** for high-performance inference
- **SLURM** for job scheduling
- **Singularity** for containerization
- **Gradio** for web-based user interface

### Target Environment
- **Primary:** KISTI Neuron GPU Cluster
- **General:** Any SLURM-managed HPC cluster with NVIDIA GPUs and Singularity support

### Key Features
1. **Dual Interface:** Gradio chat UI + OpenAI-compatible REST API
2. **HPC-Optimized:** SLURM batch script with automatic port forwarding
3. **Container-Based:** vLLM runs in Singularity container (host only needs Python for UI)
4. **Model-Agnostic:** Works with GPT-OSS, Qwen, Mistral, Llama, and other HF models
5. **Production-Ready:** Health monitoring, logging, graceful cleanup

---

## Codebase Architecture

### Component Diagram

```
┌─────────────────────────────────────────────────────────────────┐
│                         User's Laptop                           │
│  ┌──────────────────────┐       ┌────────────────────────┐    │
│  │  Web Browser         │       │  API Client (curl/SDK) │    │
│  │  http://localhost    │       │  http://localhost      │    │
│  └──────────┬───────────┘       └──────────┬─────────────┘    │
│             │ SSH Tunnel                    │ SSH Tunnel       │
└─────────────┼───────────────────────────────┼──────────────────┘
              │ :7860                         │ :8000
              │                               │
┌─────────────┼───────────────────────────────┼──────────────────┐
│             │   HPC Compute Node            │                  │
│  ┌──────────▼────────────┐     ┌───────────▼──────────────┐  │
│  │   Gradio UI           │────▶│   vLLM API Server        │  │
│  │   (vllm_web.py)       │     │   (Singularity)          │  │
│  │   Python host process │     │   /v1/chat/completions   │  │
│  │   Port: 7860          │     │   /v1/models             │  │
│  └───────────────────────┘     │   Port: 8000             │  │
│                                 └──────────┬───────────────┘  │
│                                            │                  │
│  ┌─────────────────────────────────────────▼────────────────┐ │
│  │              NVIDIA GPU (A100/H100/H200)                 │ │
│  │              Model Inference (vLLM)                      │ │
│  └──────────────────────────────────────────────────────────┘ │
│                                                                │
│  Orchestrated by: vllm_gradio_run_singularity.sh (SLURM)     │
└────────────────────────────────────────────────────────────────┘
```

### Data Flow

1. **Job Submission:** User submits SLURM job via `sbatch` or `srun`
2. **Initialization:** Script sets up environment, creates cache directories
3. **vLLM Launch:** Singularity container starts vLLM server on GPU
4. **Model Loading:** vLLM downloads/loads model weights from Hugging Face
5. **Gradio Launch:** Host-side Python starts Gradio UI, points to vLLM API
6. **Port Forwarding:** Script generates SSH tunnel command for user
7. **Request Processing:** Browser → Gradio → vLLM → GPU → Response

---

## Key Files Reference

### Core Application Files

| File | Lines | Purpose | Key Responsibilities |
|------|-------|---------|---------------------|
| `vllm_gradio_run_singularity.sh` | 374 | SLURM orchestration script | Job management, service startup, monitoring |
| `vllm_web.py` | 237 | Gradio web interface | Chat UI, API client, streaming |
| `requirements.txt` | 1 | Python dependencies | Gradio library specification |

### Documentation Files

| File | Purpose | Target Audience |
|------|---------|----------------|
| `README.md` | User-facing documentation | End users, HPC researchers |
| `AGENTS.md` | This file - developer guide | AI agents, developers |
| `agents_docs/` | Detailed technical docs | Contributors, maintainers |

### Configuration Files

| File | Purpose |
|------|---------|
| `.gitignore` | Version control exclusions |
| `LICENSE` | MIT license terms |
| `assets/` | Screenshots for documentation |

---

## Development Workflow

### Understanding the System

For detailed information about specific components:

- **Architecture:** See [agents_docs/architecture.md](agents_docs/architecture.md)
- **Shell Script:** See [agents_docs/slurm_script_guide.md](agents_docs/slurm_script_guide.md)
- **Python Application:** See [agents_docs/gradio_app_guide.md](agents_docs/gradio_app_guide.md)
- **Development Guide:** See [agents_docs/development_guide.md](agents_docs/development_guide.md)

### Making Changes

#### Pattern 1: Modifying the SLURM Script
```bash
# File: vllm_gradio_run_singularity.sh
# Use case: Changing job parameters, adding features, adjusting monitoring

# Key sections to understand:
# Lines 21-35: Default configuration variables
# Lines 54-63: CLI argument parsing
# Lines 183-204: vLLM server launch
# Lines 273-281: Gradio UI launch
# Lines 329-372: Monitoring loop

# Always test interactively first:
srun -p <partition> --gres=gpu:1 ./vllm_gradio_run_singularity.sh --model <model>
```

#### Pattern 2: Modifying the Web Interface
```bash
# File: vllm_web.py
# Use case: UI changes, API integration updates, new features

# Key sections:
# Lines 25-101: VLLMChat class (API client)
# Lines 105-225: create_interface function (Gradio UI)
# Lines 162-187: on_send function (message handling)

# Test locally (requires running vLLM server):
python vllm_web.py --host 0.0.0.0 --port 7860 --base-url http://localhost:8000/v1
```

### Code Conventions

1. **Shell Script (Bash):**
   - Use `set +e` for non-strict error handling
   - Prefix environment variables in UPPERCASE
   - Include cleanup trap for graceful shutdown
   - Add progress indicators with emoji for user feedback

2. **Python (vllm_web.py):**
   - Type hints for function parameters
   - Docstrings for complex functions
   - Use `gr.Blocks` for Gradio UI composition
   - Implement streaming for responsive chat interface

3. **Documentation:**
   - Keep README.md user-focused
   - Put technical details in agents_docs/
   - Use code examples with actual file paths
   - Include troubleshooting for common issues

---

## Detailed Documentation

The `agents_docs/` directory contains comprehensive documentation:

### Architecture & Design
- [architecture.md](agents_docs/architecture.md) - System architecture, design decisions, component interactions
- [api_reference.md](agents_docs/api_reference.md) - Complete API documentation for vLLM endpoints

### Implementation Guides
- [slurm_script_guide.md](agents_docs/slurm_script_guide.md) - Deep dive into the SLURM orchestration script
- [gradio_app_guide.md](agents_docs/gradio_app_guide.md) - Gradio application implementation details

### Development & Operations
- [development_guide.md](agents_docs/development_guide.md) - Setup, testing, contribution workflow
- [deployment_guide.md](agents_docs/deployment_guide.md) - Site-specific deployment, configuration tuning
- [troubleshooting.md](agents_docs/troubleshooting.md) - Common issues, debugging strategies

---

## Common Tasks

### Task 1: Add Support for a New Model

**Steps:**
1. Verify model compatibility with vLLM (check HuggingFace model card)
2. Update SLURM script defaults if needed (line 27-28)
3. Test with: `sbatch vllm_gradio_run_singularity.sh --model <new_model_id>`
4. Document memory requirements in README.md
5. Add to model compatibility table in documentation

**Key considerations:**
- Model size vs. GPU memory (use `--gpu-memory-utilization` to tune)
- Tensor parallelism for large models (`--tensor-parallel-size`)
- Special tokenizer requirements (most work with `--trust-remote-code`)

### Task 2: Modify Port Configuration

**Files to update:**
1. `vllm_gradio_run_singularity.sh` (lines 25-26: defaults)
2. Test with custom ports: `--vllm-port 9000 --gradio-port 7000`

**Validation:**
- Check port availability: `ss -tuln | grep <port>`
- Verify port forwarding command in generated file
- Test both Gradio UI and API endpoints

### Task 3: Add Health Check Endpoint

**Implementation location:**
- Add to `vllm_web.py` after line 224
- Use existing `VLLMChat.health()` method (line 34-39)

**Example:**
```python
def health_endpoint():
    return {"status": "healthy" if chat.health() else "unhealthy"}

# Add to Gradio interface:
gr.JSON(health_endpoint, every=10)
```

### Task 4: Improve Error Handling

**Target areas:**
1. Network errors (vllm_web.py:84-85)
2. Model loading failures (vllm_gradio_run_singularity.sh:278-283)
3. Out of memory errors (add to monitoring loop)

**Pattern:**
```python
try:
    # operation
except SpecificException as e:
    logger.error(f"Context: {e}")
    # graceful degradation or user-friendly message
```

---

## Troubleshooting Guide

### Quick Diagnostics

```bash
# Check job status
squeue -u $USER

# View logs
tail -f /scratch/$USER/gpt-oss-with-vllm-on-supercomputer/logs/vllm_server_<JOBID>.log
tail -f /scratch/$USER/gpt-oss-with-vllm-on-supercomputer/logs/gradio_server_<JOBID>.log

# Check GPU availability
nvidia-smi

# Test vLLM API (from compute node)
curl http://localhost:8000/v1/models

# Test Gradio UI (from compute node)
curl http://localhost:7860
```

### Common Issues

1. **"vLLM API not responding"**
   - Check GPU memory: `nvidia-smi`
   - Review vLLM log for errors
   - Try smaller model or adjust `--gpu-memory-utilization`

2. **"Port already in use"**
   - Use custom ports: `--vllm-port 9000 --gradio-port 7000`
   - Kill existing processes: `pkill -f "vllm serve"`

3. **"Model download fails"**
   - Check network connectivity from compute nodes
   - Verify HF_HOME has sufficient space: `df -h $HF_HOME`
   - Set HuggingFace token if private model: `export HF_TOKEN=...`

4. **"Gradio shows 'No models available'"**
   - Wait for vLLM to finish loading (check logs)
   - Click "Refresh Models" button
   - Verify vLLM API is running: `curl localhost:8000/v1/models`

For comprehensive troubleshooting, see [agents_docs/troubleshooting.md](agents_docs/troubleshooting.md).

---

## File Paths Quick Reference

All paths in this repository are relative to:
```
/Users/lsetiawan/Repos/SSEC/gpt-oss-with-vllm-on-supercomputer
```

For HPC deployment, paths are typically:
```
/scratch/$USER/gpt-oss-with-vllm-on-supercomputer
```

Key runtime directories:
- **Model cache:** `$HF_HOME/hub` (default: `/scratch/$USER/.huggingface/hub`)
- **vLLM cache:** `/scratch/$USER/gpt-oss-with-vllm-on-supercomputer/.vllm`
- **Logs:** `/scratch/$USER/gpt-oss-with-vllm-on-supercomputer/logs/`
- **Singularity image:** `/scratch/$USER/gpt-oss-with-vllm-on-supercomputer/vllm-gptoss.sif`

---

## Project Status & Roadmap

### Current Version
- Stable release for KISTI Neuron GPU Cluster
- Tested with GPT-OSS 20B, Qwen, Mistral models
- Single-node, multi-GPU support via tensor parallelism

### Known Limitations
- Manual SSH port forwarding setup (could automate)
- No persistent conversation storage (in-memory only)
- Requires manual model selection before job submission

### Potential Improvements
- Add conversation history persistence
- Implement automatic SSH tunnel establishment
- Support multi-node distributed inference
- Add user authentication for web interface
- Create model performance benchmarking tools

For contributing ideas or implementations, see [agents_docs/development_guide.md](agents_docs/development_guide.md).

---

## Additional Resources

- **vLLM Documentation:** https://github.com/vllm-project/vllm
- **Gradio Documentation:** https://www.gradio.app/docs
- **SLURM Documentation:** https://slurm.schedmd.com/documentation.html
- **Singularity Documentation:** https://docs.sylabs.io/guides/3.5/user-guide/

---

## Getting Help

1. Check [troubleshooting.md](agents_docs/troubleshooting.md) for common issues
2. Review log files for specific error messages
3. Consult SLURM job output: `cat slurm-<JOBID>.out`
4. Open GitHub issue with logs and environment details

---

**Last Updated:** 2025-01-27

**Maintainer:** Soonwook Hwang (https://github.com/hwang2006/gpt-oss-with-vllm-on-supercomputer)
