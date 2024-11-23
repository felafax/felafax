#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

if [ "${1:-format}" = "check" ]; then
    ruff check ./src/felafax/ ./trainers/ --config "${base_dir}/pyproject.toml"
else
    ruff format ./src/felafax/ ./trainers/ --config "${base_dir}/pyproject.toml"
fi