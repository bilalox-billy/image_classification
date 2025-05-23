#!/bin/bash

PYTHON_VERSION=3.11

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
VENV_DIR=$DIR/venv

# Create virtual using specific pyenv version
yes no | pyenv install $PYTHON_VERSION
pyenv local $PYTHON_VERSION

# Delete virtual env if it exists
rm -rf $VENV_DIR

# Create virtual env
PYENV_PYTHON=$(pyenv which python)
$PYENV_PYTHON -m venv $VENV_DIR
source $VENV_DIR/Scripts/activate



# Install dependencies
poetry install --no-root
