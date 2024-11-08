#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

ruff check . --fix --config "${base_dir}/pyproject.toml" 
ruff format . --config "${base_dir}/pyproject.toml"