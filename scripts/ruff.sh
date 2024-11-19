#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

if [ "${1:-format}" = "check" ]; then
    ruff check ./felafax/ ./projects/ --config "${base_dir}/pyproject.toml"
else
    ruff format ./felafax/ ./projects/ --config "${base_dir}/pyproject.toml"
fi