#!/usr/bin/env bash
set -euo pipefail

if [[ ! -d ".venv" ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Offline setup complete."
echo "Next:"
echo "1) streamlit run app.py"
echo "2) Click 'Build Knowledge Base Index'"
