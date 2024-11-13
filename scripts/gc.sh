#!/bin/bash
set -Eeuo pipefail

base_dir=$(dirname $(dirname $0))

# Get commit message from command line argument
if [ $# -eq 0 ]; then
    echo "Error: Please provide a commit message"
    exit 1
fi

commit_msg="$1"

# Run git commit and push
git commit -m "$commit_msg"
git push
