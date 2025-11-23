#!/bin/bash

# Run flake8 with specified rules
flake8 . --count --max-complexity=12 --max-line-length=100 \
    --exclude='./script/*' \
    --per-file-ignores="__init__.py:F401" \
    --statistics