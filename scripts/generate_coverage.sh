#!/usr/bin/env bash
# Generate coverage report and badge
coverage run -m pytest
coverage xml -i
coverage-badge -o coverage.svg -f
