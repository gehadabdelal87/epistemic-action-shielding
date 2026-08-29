#!/bin/bash
set -e
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"

echo "EAS/DQN project: $ROOT"
echo "Python: $(python3 --version 2>&1)"

if [ ! -d .venv ]; then
  python3 -m venv .venv
fi
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install -e '.[experiments,test]'
python -m pytest

echo
echo "Setup complete."
echo "Quick DQN run:"
echo "  source .venv/bin/activate"
echo "  python run_dqn_experiment.py --training-episodes 500 --evaluation-episodes 20 --training-seeds 2026 --output results/dqn_quick"
echo
echo "Full configured run:"
echo "  bash scripts/reproduce.sh configs/journal.json results/journal"
