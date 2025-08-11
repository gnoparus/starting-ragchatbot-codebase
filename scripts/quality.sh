#!/bin/bash

# Code Quality Check Script
# This script runs all code quality tools in sequence

set -e

echo "🔍 Running code quality checks..."

echo "📝 Running Black formatter..."
uv run black --check backend/ main.py

echo "📦 Checking import sorting with isort..."
uv run isort --check-only backend/ main.py

echo "🔧 Running flake8 linter..."
uv run flake8 backend/ main.py

echo "🔬 Running mypy type checker..."
uv run mypy backend/ main.py

echo "✅ All code quality checks passed!"