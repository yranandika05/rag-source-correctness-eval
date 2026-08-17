#!/usr/bin/env bash
set -e

mkdir -p data

echo "Fetching Github Docs ..."
if [ ! -d "data/github_docs" ]; then
    git clone --depth 1 --filter=blob:none --sparse https://github.com/github/docs.git data/github_docs
    git -C data/github_docs sparse-checkout set content
else
    echo "Github Docs already exists, skipping clone."
fi

echo "Fetching Gitlab Docs ..."
if [ ! -d "data/gitlab_docs" ]; then
    git clone --depth 1 --filter=blob:none --sparse https://gitlab.com/gitlab-org/gitlab.git data/gitlab_docs
    git -C data/gitlab_docs sparse-checkout set doc
else
    echo "Gitlab Docs already exists, skipping clone."
fi

echo "Done."

echo "Documentation repositories are stored locally under data/ and should be ignored by Git."
