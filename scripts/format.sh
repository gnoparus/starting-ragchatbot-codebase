#!/bin/bash

# Code Formatting Script
# This script formats all code using black and isort

set -e

echo "🎨 Formatting code..."

echo "📝 Running Black formatter..."
uv run black backend/ main.py

echo "📦 Sorting imports with isort..."
uv run isort backend/ main.py

echo "✅ Code formatting complete!"