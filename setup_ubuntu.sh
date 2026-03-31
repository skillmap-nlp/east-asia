#!/usr/bin/env bash
# Setup script for Ubuntu GPU server — Gemma-3 translation with vLLM
# Run once before first use:   bash setup_ubuntu.sh

set -e

echo "=== Installing system packages ==="
sudo apt-get update -q
sudo apt-get install -y python3-pip python3-venv git

echo ""
echo "=== Creating virtual environment ==="
python3 -m venv venv
source venv/bin/activate

echo ""
echo "=== Installing Python packages ==="
pip install --upgrade pip
pip install "vllm>=0.4.0"
pip install "transformers>=4.50.0"
pip install accelerate huggingface_hub

echo ""
echo "=== Logging in to HuggingFace (needed for Gemma gated model) ==="
echo "  Run manually:  huggingface-cli login"
echo "  Then paste your token from https://huggingface.co/settings/tokens"

echo ""
echo "=== Done. Activate env and run: ==="
echo "  source venv/bin/activate"
echo "  python translate_gemma_vllm.py --dry-run     # check counts"
echo "  python translate_gemma_vllm.py               # run all tables"
echo "  python translate_gemma_vllm.py --table jobads_jp   # one table"
echo "  python translate_gemma_vllm.py --apply-only  # write checkpoints → DB"
