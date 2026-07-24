#!/bin/bash
# Pre-commit hook to run pytest
# This script ensures all tests pass before allowing commits

set -e

echo "Running test suite..."
echo "====================="

# Change to project root
cd "$(git rev-parse --show-toplevel)"

# Run pytest with required coverage
python -m pytest tests/ -v --tb=line --cov=src --cov-report=term-missing --cov-fail-under=80

if [ $? -eq 0 ]; then
    echo ""
    echo "✓ All tests passed!"
    exit 0
else
    echo ""
    echo "✗ Tests failed. Please fix before committing."
    exit 1
fi
