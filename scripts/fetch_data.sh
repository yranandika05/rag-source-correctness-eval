#!/usr/bin/env bash
set -e

mkdir -p data

echo "Fetching Github Docs ..."
if [ ! -d "data/github_docs" ]; then
    git clone --depth 1 https://github.com/github/docs.git data/github_docs
else
    echo "Github Docs already exists, skipping clone."
fi

echo "Fetching Gitlab Docs ..."
if [ ! -d "data/gitlab_docs" ]; then
    git clone --depth 1 https://gitlab.com/gitlab-org/technical-writing/docs-gitlab-com.git data/gitlab_docs
else
    echo "Gitlab Docs already exists, skipping clone."
fi

echo "Done."

echo "Documentation repositories are stored locally under data/ and should be ignored by Git."