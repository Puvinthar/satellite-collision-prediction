#!/bin/bash
set -e

MAMBA_ROOT_PREFIX="$HOME/.mamba"
MAMBA_EXE="$HOME/.local/bin/micromamba"
ENV_PATH="$(pwd)/.venv"

if [ ! -f "$MAMBA_EXE" ]; then
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj bin/micromamba
    mkdir -p "$HOME/.local/bin"
    mv bin/micromamba "$MAMBA_EXE"
    rmdir bin || true
    chmod +x "$MAMBA_EXE"
fi

export MAMBA_ROOT_PREFIX
eval "$("$MAMBA_EXE" shell hook -s bash)"

# Create the environment directly in the project .venv directory
"$MAMBA_EXE" create -p "$ENV_PATH" python=3.12 -c conda-forge -c tudat-team tudatpy -y

# Activate via the path
micromamba activate "$ENV_PATH"

# uv automatically detects the .venv directory
uv sync

uv run python -c "import tudatpy; print(tudatpy.__version__)"