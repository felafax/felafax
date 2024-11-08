#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))
ruff check . --config "${base_dir}/pyproject.toml"
