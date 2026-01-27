# Deployment Guide

> Site-specific deployment, configuration, and production considerations

---

## Table of Contents

1. [Overview](#overview)
2. [Cluster-Specific Customization](#cluster-specific-customization)
3. [Production Configuration](#production-configuration)
4. [Multi-User Deployment](#multi-user-deployment)
5. [Performance Optimization](#performance-optimization)
6. [Monitoring & Maintenance](#monitoring--maintenance)
7. [Backup & Recovery](#backup--recovery)
8. [Security Hardening](#security-hardening)

---

## Overview

This guide helps system administrators and power users deploy this solution on different HPC clusters with appropriate customizations for production use.

**Deployment Scenarios:**
1. **Single User Development:** Default setup, minimal changes
2. **Research Group:** Shared models, collaborative access
3. **Production Service:** High availability, monitoring, quotas
4. **Multi-Cluster:** Portable configuration across sites

---

## Cluster-Specific Customization

### Template: New Cluster Deployment

**Step 1: Identify Cluster Parameters**

Create a cluster profile file: `/scratch/$USER/cluster-config.sh`

```bash
# KISTI Neuron (example)
export CLUSTER_NAME="kisti-neuron"
export CLUSTER_LOGIN="neuron.ksc.re.kr"
export CLUSTER_MODULES="gcc/10.2.0 cuda/12.1"
export CLUSTER_PARTITION="amd_a100nv_8"
export CLUSTER_SCRATCH="/scratch"
export CLUSTER_CONDA="/scratch/$USER/miniconda3"
```

**Step 2: Customize SLURM Script**

Edit `vllm_gradio_run_singularity.sh`:

```bash
# Lines 4-17: Update SLURM directives
#SBATCH --partition=${CLUSTER_PARTITION}
# Add cluster-specific flags:
#SBATCH --account=<your-account>  # If required
#SBATCH --qos=<qos-name>          # If quality-of-service used

# Lines 133-134: Update module loads
module load ${CLUSTER_MODULES}

# Line 128: Update login node in port forwarding
echo "ssh ... ${USER}@${CLUSTER_LOGIN}" > "$PORT_FWD_FILE"
```

**Step 3: Test Deployment**

```bash
# Interactive test
srun -p ${CLUSTER_PARTITION} --gres=gpu:1 --time=1:00:00 --pty bash
source /scratch/$USER/cluster-config.sh
cd /scratch/$USER/gpt-oss-with-vllm-on-supercomputer
./vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B

# Verify:
# - Modules load without errors
# - GPU detected (nvidia-smi)
# - Model downloads successfully
# - Services start and respond
```

---

### Example 1: NCSA Delta Cluster

**Profile:**
```bash
# /scratch/$USER/delta-config.sh
export CLUSTER_NAME="ncsa-delta"
export CLUSTER_LOGIN="login.delta.ncsa.illinois.edu"
export CLUSTER_MODULES="gcc/11.2.0 cuda/11.8.0"
export CLUSTER_PARTITION="gpuA100x4"
export CLUSTER_SCRATCH="/scratch/bbka"
```

**Customizations:**
```bash
# SLURM script changes
#SBATCH --partition=gpuA100x4
#SBATCH --account=bbka-delta-gpu
#SBATCH --mem=64G  # Delta requires explicit memory request

# Module loads (line 134)
module load gcc/11.2.0 cuda/11.8.0 anaconda3_gpu

# Conda path (line 146)
if [ -f "/sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh" ]; then
  . "/sw/external/python/anaconda3_gpu/etc/profile.d/conda.sh"
fi
```

---

### Example 2: TACC Frontera (No Singularity)

**Alternative: Native vLLM Installation**

If Singularity not available, use conda/pip:

```bash
# Create environment with vLLM
module load gcc/9.1.0 cuda/12.0 python3/3.9.7
conda create -n vllm-env python=3.10
conda activate vllm-env
pip install vllm==0.10.2 gradio==5.42.0

# Modify script to run vLLM directly (no Singularity)
# Replace line 190-204 with:
nohup vllm serve "${VLLM_MODEL}" \
  --host 0.0.0.0 \
  --port "${VLLM_PORT}" \
  # ... other flags ...
  > "$VLLM_LOG" 2>&1 &
```

---

### Example 3: AWS ParallelCluster

**Cloud HPC Deployment:**

```bash
# ParallelCluster configuration (pcluster.yaml)
HeadNode:
  InstanceType: t3.xlarge
Scheduling:
  Scheduler: slurm
  SlurmQueues:
    - Name: gpu
      CapacityType: ONDEMAND
      Networking:
        SubnetIds: [subnet-xxx]
      ComputeResources:
        - Name: gpu-nodes
          InstanceType: p4d.24xlarge  # 8x A100 GPUs
          MinCount: 0
          MaxCount: 4

# Deployment
pcluster create-cluster --cluster-name vllm-cluster --cluster-configuration pcluster.yaml

# Usage (same as HPC)
pcluster ssh vllm-cluster
cd /shared
git clone https://github.com/hwang2006/gpt-oss-with-vllm-on-supercomputer.git
# ... rest of setup ...
```

---

## Production Configuration

### High-Availability Setup

**Goal:** Minimize downtime, handle failures gracefully

**1. Automatic Restart on Crash**

Add to monitoring loop (line 345):

```bash
# If vLLM dies, restart it
if ! kill -0 "$VLLM_PID" 2>/dev/null; then
  echo "[$(date)] ⚠️  vLLM died, restarting..."

  # Relaunch vLLM (copy lines 190-207)
  nohup singularity exec --nv "$SIF_PATH" \
    vllm serve "${VLLM_MODEL}" \
    # ... flags ...
    > "$VLLM_LOG" 2>&1 &
  VLLM_PID=$!

  # Wait for readiness
  sleep 10
  until curl -fsS "http://127.0.0.1:${VLLM_PORT}/v1/models" >/dev/null 2>&1; do
    sleep 5
  done
  echo "[$(date)] ✅ vLLM restarted successfully"
fi
```

**2. Persistent Job (Batch Job with Resubmission)**

Create wrapper script: `persistent_vllm.sh`

```bash
#!/bin/bash
JOBID=$(sbatch --parsable vllm_gradio_run_singularity.sh)
echo "Submitted job: $JOBID"

# Monitor job status
while true; do
  STATUS=$(sacct -j $JOBID --format=State --noheader | tail -1 | tr -d ' ')

  if [[ "$STATUS" == "COMPLETED" || "$STATUS" == "FAILED" ]]; then
    echo "Job $JOBID ended with status: $STATUS"

    # Resubmit if failed
    if [ "$STATUS" == "FAILED" ]; then
      echo "Resubmitting..."
      JOBID=$(sbatch --parsable vllm_gradio_run_singularity.sh)
      echo "New job: $JOBID"
    fi
  fi

  sleep 60
done
```

**3. Load Balancer (Multiple Instances)**

Run multiple jobs on different ports:

```bash
# Job 1: Ports 8000, 7860
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b --vllm-port 8000 --gradio-port 7860

# Job 2: Ports 8001, 7861
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b --vllm-port 8001 --gradio-port 7861

# Job 3: Ports 8002, 7862
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b --vllm-port 8002 --gradio-port 7862

# Use nginx/HAProxy to load balance across 8000-8002
```

---

### Resource Quotas & Limits

**Prevent Resource Exhaustion:**

```bash
# Limit max tokens per request (add to vLLM flags)
--max-model-len 8192  # Cap context window

# Limit concurrent requests
export MAX_NUM_SEQS=32  # Prevents overload

# SLURM cgroup limits (in job script)
#SBATCH --mem=100G  # Cap RAM usage
#SBATCH --gres=gpu:1  # Explicit GPU count

# Add per-user rate limiting (in Gradio app)
# Requires modification to vllm_web.py:
from collections import defaultdict
import time

request_counts = defaultdict(list)
RATE_LIMIT = 60  # requests per minute

def rate_limit_check(user_ip):
    now = time.time()
    request_counts[user_ip] = [t for t in request_counts[user_ip] if now - t < 60]

    if len(request_counts[user_ip]) >= RATE_LIMIT:
        raise gr.Error("Rate limit exceeded. Please wait.")

    request_counts[user_ip].append(now)
```

---

### Centralized Configuration

**Create shared configuration file:**

`/shared/vllm-config/production.env`

```bash
# Model settings
export VLLM_MODEL="openai/gpt-oss-20b"
export MAX_MODEL_LEN=4096
export GPU_MEM_UTIL=0.85

# Network settings
export VLLM_PORT=8000
export GRADIO_PORT=7860

# Paths
export HF_HOME="/shared/models/.huggingface"
export SIF_PATH="/shared/containers/vllm-gptoss.sif"

# Performance tuning
export MAX_NUM_SEQS=32
export TP_SIZE=1
```

**Use in job script:**

```bash
# Add after line 40
if [ -f "/shared/vllm-config/production.env" ]; then
  source "/shared/vllm-config/production.env"
fi
```

---

## Multi-User Deployment

### Shared Model Cache

**Benefits:**
- Single model download (saves bandwidth and time)
- Consistent versions across users
- Reduced storage usage

**Setup:**

```bash
# Create shared directory (admin)
sudo mkdir -p /shared/models/.huggingface/hub
sudo chmod 775 /shared/models/.huggingface
sudo chgrp research-group /shared/models/.huggingface

# Pre-download models (admin or first user)
export HF_HOME="/shared/models/.huggingface"
python -c "
from huggingface_hub import snapshot_download
snapshot_download('openai/gpt-oss-20b', cache_dir='$HF_HOME/hub')
snapshot_download('Qwen/Qwen3-0.6B', cache_dir='$HF_HOME/hub')
"

# Users set environment variable
export HF_HOME="/shared/models/.huggingface"
sbatch vllm_gradio_run_singularity.sh --model openai/gpt-oss-20b
```

---

### Port Management

**Avoid conflicts with multiple users:**

**Option A: Dynamic Port Allocation**

```bash
# Add to script (after line 25)
# Find available ports
find_free_port() {
  local start_port=$1
  local port=$start_port
  while ss -tuln | grep -q ":$port "; do
    port=$((port + 1))
  done
  echo $port
}

VLLM_PORT=$(find_free_port 8000)
GRADIO_PORT=$(find_free_port 7860)
echo "Allocated ports: vLLM=$VLLM_PORT, Gradio=$GRADIO_PORT"
```

**Option B: User-Specific Ports**

```bash
# Assign port ranges per user
# User alice: 8000-8009, 7860-7869
# User bob:   8010-8019, 7870-7879

USER_ID=$(id -u)
PORT_OFFSET=$((USER_ID % 100 * 10))
VLLM_PORT=$((8000 + PORT_OFFSET))
GRADIO_PORT=$((7860 + PORT_OFFSET))
```

---

### Access Control

**Option A: Network-Level (Firewall)**

```bash
# Only allow access from specific subnets
iptables -A INPUT -p tcp --dport 7860:7869 -s 10.0.0.0/8 -j ACCEPT
iptables -A INPUT -p tcp --dport 7860:7869 -j DROP
```

**Option B: Application-Level (Gradio Auth)**

```python
# Modify vllm_web.py launch (line 236)
ui.launch(
    server_name=args.host,
    server_port=args.port,
    auth=("username", "password"),  # Simple auth
    # Or use auth_message and auth callback for custom logic
)
```

---

## Performance Optimization

### Model-Specific Tuning

**Small Models (< 10B parameters):**
```bash
export MAX_MODEL_LEN=8192
export GPU_MEM_UTIL=0.95
export MAX_NUM_SEQS=128
```

**Medium Models (10-30B parameters):**
```bash
export MAX_MODEL_LEN=4096
export GPU_MEM_UTIL=0.90
export MAX_NUM_SEQS=64
```

**Large Models (34-70B parameters):**
```bash
#SBATCH --gres=gpu:2  # Use 2+ GPUs
export TP_SIZE=2
export MAX_MODEL_LEN=2048
export GPU_MEM_UTIL=0.85
export MAX_NUM_SEQS=16
```

---

### Advanced vLLM Flags

**Enable prefix caching (for repeated prompts):**
```bash
# Add to vllm serve command
--enable-prefix-caching
# Speeds up: chatbots with fixed system prompts, batch processing
```

**Chunked prefill (reduce first-token latency):**
```bash
--enable-chunked-prefill
--max-num-batched-tokens 8192
# Trade-off: Slightly lower throughput for faster first token
```

**Speculative decoding (experimental):**
```bash
--speculative-model <small-model>
--num-speculative-tokens 5
# Uses smaller model to predict tokens, verified by main model
# Can improve throughput by 20-50% for certain workloads
```

---

### GPU Optimization

**Use MIG (Multi-Instance GPU) for A100/H100:**

```bash
# Partition A100 into smaller instances
sudo nvidia-smi mig -cgi 9,9,9,9  # 4x 10GB instances

# Allocate MIG instance to job
#SBATCH --gres=gpu:1g.10gb

# Multiple users can share same physical GPU
```

**Enable GPU persistence mode:**

```bash
# Reduces GPU initialization time
sudo nvidia-smi -pm 1
```

---

## Monitoring & Maintenance

### Centralized Logging

**Aggregate logs to central location:**

```bash
# Add to script (after line 81)
CENTRAL_LOG_DIR="/shared/logs/vllm"
mkdir -p "$CENTRAL_LOG_DIR"

# Symlink logs
ln -sf "$VLLM_LOG" "$CENTRAL_LOG_DIR/vllm_${USER}_${JOB_ID}.log"
ln -sf "$GRADIO_LOG" "$CENTRAL_LOG_DIR/gradio_${USER}_${JOB_ID}.log"
```

**Log rotation:**

```bash
# Weekly cron job (on login node)
0 0 * * 0 find /shared/logs/vllm -name "*.log" -mtime +30 -delete
```

---

### Prometheus Metrics

**vLLM exposes metrics endpoint:**

```bash
curl http://localhost:8000/metrics
# Output: Prometheus-format metrics
# - vllm:num_requests_running
# - vllm:num_requests_waiting
# - vllm:gpu_cache_usage_perc
# - vllm:avg_prompt_throughput_toks_per_s
```

**Scrape with Prometheus:**

```yaml
# prometheus.yml
scrape_configs:
  - job_name: 'vllm'
    static_configs:
      - targets: ['gpu05:8000', 'gpu06:8000']
    metrics_path: /metrics
```

**Create Grafana dashboard:**
- Import vLLM dashboard template
- Visualize: throughput, latency, queue length, GPU utilization

---

### Health Monitoring Script

**Create monitoring daemon:**

`/shared/scripts/monitor_vllm.sh`

```bash
#!/bin/bash
while true; do
  for port in 8000 8001 8002; do
    if curl -sf "http://localhost:${port}/v1/models" >/dev/null; then
      echo "$(date) [OK] vLLM on port $port is healthy"
    else
      echo "$(date) [FAIL] vLLM on port $port is down!"
      # Send alert (email, Slack, etc.)
      # Attempt restart if auto-healing enabled
    fi
  done
  sleep 60
done
```

---

### Automated Model Updates

**Check for model updates and reload:**

```bash
#!/bin/bash
# /shared/scripts/update_models.sh

MODEL="openai/gpt-oss-20b"
CACHE_DIR="/shared/models/.huggingface/hub"

# Check HuggingFace for new commits
CURRENT_REV=$(cat "$CACHE_DIR/$MODEL/current_revision.txt" 2>/dev/null || echo "unknown")
LATEST_REV=$(curl -sL "https://huggingface.co/api/models/$MODEL" | jq -r '.sha')

if [ "$CURRENT_REV" != "$LATEST_REV" ]; then
  echo "New model version detected: $LATEST_REV"

  # Download new version
  python -c "
  from huggingface_hub import snapshot_download
  snapshot_download('$MODEL', cache_dir='$CACHE_DIR')
  "

  echo "$LATEST_REV" > "$CACHE_DIR/$MODEL/current_revision.txt"

  # Notify users to restart jobs
  wall "New version of $MODEL available. Please restart your vLLM jobs."
fi
```

---

## Backup & Recovery

### Critical Data

**What to back up:**
1. **Model cache:** `/shared/models/.huggingface` (large, but critical)
2. **Configuration:** Job scripts, cluster configs
3. **Logs:** For audit and debugging
4. **Conversation histories:** If persistence added

**What NOT to back up:**
- vLLM compilation cache (`.vllm/`) - regenerates automatically
- Temporary files (`/tmp`, `/scratch/tmp`)
- Singularity SIF files (rebuild from Docker)

---

### Backup Strategy

**Incremental backup with rsync:**

```bash
#!/bin/bash
# Daily backup script

SOURCE="/shared/models/.huggingface"
DEST="/backup/huggingface-$(date +%Y%m%d)"

rsync -avz --link-dest=/backup/huggingface-latest \
  "$SOURCE" "$DEST"

ln -sfn "$DEST" /backup/huggingface-latest
```

**Model verification:**

```bash
# After backup, verify integrity
cd "$DEST"
find . -name "*.bin" -o -name "*.safetensors" | while read file; do
  md5sum "$file" >> checksums.txt
done

# Compare with source
diff checksums.txt "$SOURCE/checksums.txt"
```

---

### Disaster Recovery

**Scenario: Complete data loss**

**Recovery steps:**

1. **Reinstall base system:**
   ```bash
   git clone https://github.com/hwang2006/gpt-oss-with-vllm-on-supercomputer.git
   cd gpt-oss-with-vllm-on-supercomputer
   ```

2. **Rebuild container:**
   ```bash
   singularity build --fakeroot vllm-gptoss.sif docker://vllm/vllm-openai:gptoss
   ```

3. **Restore model cache (from backup):**
   ```bash
   rsync -avz /backup/huggingface-latest/ /shared/models/.huggingface/
   ```

4. **Restore configuration:**
   ```bash
   cp /backup/configs/*.env /shared/vllm-config/
   ```

5. **Test deployment:**
   ```bash
   sbatch vllm_gradio_run_singularity.sh --model Qwen/Qwen3-0.6B
   ```

**RTO (Recovery Time Objective):** ~2 hours (container rebuild + config restore)
**RPO (Recovery Point Objective):** 24 hours (daily backup)

---

## Security Hardening

### Network Security

**1. Restrict port binding:**

```bash
# Bind only to localhost (not 0.0.0.0)
# Modify line 192 in script
--host 127.0.0.1  # Instead of 0.0.0.0

# Access via SSH tunnel only (no direct access)
```

**2. Use SSH key authentication only:**

```bash
# On cluster, disable password auth
# /etc/ssh/sshd_config
PasswordAuthentication no
PubkeyAuthentication yes
```

**3. IP whitelist:**

```bash
# In job script, check source IP
ALLOWED_IPS="10.0.0.0/8 192.168.1.0/24"
if ! echo "$ALLOWED_IPS" | grep -q "$(echo $SSH_CLIENT | awk '{print $1}')"; then
  echo "Access denied from $(echo $SSH_CLIENT | awk '{print $1}')"
  exit 1
fi
```

---

### Container Security

**1. Use read-only SIF:**

```bash
# SIF files are already read-only by design
# Verify
ls -l vllm-gptoss.sif
# Should show: -r--r--r-- (no write permissions)
```

**2. Scan container for vulnerabilities:**

```bash
# Convert SIF to Docker for scanning
singularity build --docker-login vllm-docker.tar vllm-gptoss.sif

# Scan with Trivy
trivy image --input vllm-docker.tar
```

**3. Use signed containers:**

```bash
# Sign SIF with GPG key (admin)
singularity sign vllm-gptoss.sif

# Verify before use (users)
singularity verify vllm-gptoss.sif
```

---

### Data Security

**1. Encrypt sensitive data:**

```bash
# Encrypt conversation history (if added)
# Use age or GPG
age -r <public-key> -o conversation.json.age conversation.json
```

**2. Sanitize logs:**

```bash
# Remove sensitive data from logs before archiving
sed -i 's/[A-Za-z0-9._%+-]\+@[A-Za-z0-9.-]\+\.[A-Z|a-z]\{2,\}/[EMAIL]/g' vllm_server.log
```

**3. Implement audit logging:**

```bash
# Log all requests
# Add to vllm_web.py:
import logging
logging.basicConfig(filename='audit.log', level=logging.INFO)

def on_send(msg, hist, ...):
    logging.info(f"User {user_id} sent: {msg[:50]}...")  # Log first 50 chars
    # ... rest of function
```

---

## Summary

This deployment guide provides:
- **Cluster-specific customization** for different HPC sites
- **Production configuration** for reliability and performance
- **Multi-user deployment** strategies
- **Performance optimization** techniques
- **Monitoring & maintenance** best practices
- **Backup & recovery** procedures
- **Security hardening** measures

**Key Takeaways:**
- Adapt SLURM directives and module loads for each cluster
- Use shared model cache to save resources
- Implement monitoring and alerting for production
- Secure deployments with network restrictions and authentication
- Regular backups of critical data (models, configs, logs)

**Next Steps:**
1. Create cluster-specific configuration
2. Test with small model first
3. Implement monitoring before production
4. Document site-specific quirks
5. Train users on access procedures

---

**Document Version:** 1.0
**Last Updated:** 2025-01-27
