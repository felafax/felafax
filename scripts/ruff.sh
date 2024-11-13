#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

if [ "${1:-format}" = "check" ]; then
    ruff check . --config "${base_dir}/pyproject.toml"
else
    ruff format . --config "${base_dir}/pyproject.toml"
fi