# Technical Documentation Directory

This directory contains comprehensive technical documentation for the GPT-OSS with vLLM on Supercomputer project.

## Documentation Index

### Getting Started
- **[../AGENTS.md](../AGENTS.md)** - Main entry point for AI agents and developers. Start here!

### Architecture & Design
- **[architecture.md](architecture.md)** - Complete system architecture, component interactions, design decisions, and performance characteristics

### Implementation Guides
- **[slurm_script_guide.md](slurm_script_guide.md)** - Deep dive into `vllm_gradio_run_singularity.sh` (374 lines of orchestration logic)
- **[gradio_app_guide.md](gradio_app_guide.md)** - Complete analysis of `vllm_web.py` (Gradio web interface implementation)

### API Reference
- **[api_reference.md](api_reference.md)** - Full documentation of vLLM OpenAI-compatible API endpoints with examples

### Operations
- **[development_guide.md](development_guide.md)** - Setup, testing, code conventions, and contributing workflow
- **[deployment_guide.md](deployment_guide.md)** - Site-specific deployment, production configuration, multi-user setup
- **[troubleshooting.md](troubleshooting.md)** - Common issues, diagnostic procedures, and debugging strategies

## Documentation Structure

Each guide is designed for progressive disclosure:
1. **Quick reference** at the top (TL;DR)
2. **Conceptual overview** of the topic
3. **Detailed analysis** with code examples
4. **Practical examples** and common tasks
5. **Troubleshooting** and edge cases

## Reading Paths

### For AI Coding Agents
1. Start with [../AGENTS.md](../AGENTS.md) - overview and file reference
2. Review [architecture.md](architecture.md) - understand system design
3. Deep dive into specific component guides as needed
4. Consult [api_reference.md](api_reference.md) for API details
5. Use [troubleshooting.md](troubleshooting.md) when debugging

### For Human Developers
1. Read [../AGENTS.md](../AGENTS.md) for orientation
2. Follow [development_guide.md](development_guide.md) to set up environment
3. Explore implementation guides for components you'll modify
4. Reference [troubleshooting.md](troubleshooting.md) when issues arise

### For System Administrators
1. Review [architecture.md](architecture.md) - understand deployment context
2. Follow [deployment_guide.md](deployment_guide.md) for site customization
3. Implement monitoring from operations guides
4. Keep [troubleshooting.md](troubleshooting.md) handy for user support

### For Researchers/Users
1. Start with [../README.md](../README.md) (user-facing documentation)
2. Consult [api_reference.md](api_reference.md) for API usage
3. Check [troubleshooting.md](troubleshooting.md) if problems occur

## Documentation Statistics

| Document | Lines | Words | Focus Area |
|----------|-------|-------|------------|
| architecture.md | ~1000 | ~7000 | System design, components, data flow |
| slurm_script_guide.md | ~900 | ~6500 | SLURM orchestration script analysis |
| gradio_app_guide.md | ~700 | ~5000 | Web interface implementation |
| api_reference.md | ~600 | ~4000 | vLLM API endpoints and usage |
| development_guide.md | ~850 | ~6000 | Setup, testing, contributing |
| deployment_guide.md | ~800 | ~5500 | Production deployment strategies |
| troubleshooting.md | ~900 | ~6500 | Issue diagnosis and resolution |
| **Total** | **~5750** | **~40500** | **Comprehensive coverage** |

## Maintenance

### Updating Documentation

When making code changes, update relevant documentation:

**Code Change → Documentation Update:**
- Modify `vllm_gradio_run_singularity.sh` → Update `slurm_script_guide.md`
- Modify `vllm_web.py` → Update `gradio_app_guide.md`
- Add new feature → Update `AGENTS.md` and relevant guide
- Change API behavior → Update `api_reference.md`
- New deployment scenario → Update `deployment_guide.md`

### Documentation Standards

- **Accuracy:** Code examples must work as written
- **Completeness:** Cover both success and failure cases
- **Currency:** Update version numbers and dates
- **Clarity:** Write for diverse skill levels
- **Examples:** Include real, tested examples

## Contributing to Documentation

1. **Small fixes:** Edit directly and commit
2. **New sections:** Follow existing structure and style
3. **Major changes:** Discuss in issue first

### Style Guide

- Use Markdown with consistent formatting
- Include code blocks with language tags
- Add file paths with absolute references
- Use diagrams (ASCII art) for complex flows
- Provide both conceptual and practical content

## Questions & Feedback

- **GitHub Issues:** Report doc issues or suggest improvements
- **Pull Requests:** Contribute documentation enhancements
- **Discussion:** Ask questions in GitHub Discussions

---

**Last Updated:** 2025-01-27
**Maintainer:** Project Documentation Team
