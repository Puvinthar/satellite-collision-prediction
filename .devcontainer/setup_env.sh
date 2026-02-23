#!/bin/bash
set -e

export MAMBA_EXE="$HOME/.local/bin/micromamba"
export MAMBA_ROOT_PREFIX="${MAMBA_ROOT_PREFIX:-$HOME/.mamba}"
ENV_PATH="$(pwd)/.venv"
CURRENT_SHELL=$(basename "${SHELL:-bash}")
RC_FILE="$HOME/.${CURRENT_SHELL}rc"


if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"
fi

if [ ! -f "$MAMBA_EXE" ]; then
    mkdir -p "$HOME/.local/bin"
    curl -Ls https://micro.mamba.pm/api/micromamba/linux-64/latest | tar -xvj -O bin/micromamba > "$MAMBA_EXE"
    chmod +x "$MAMBA_EXE"
fi

# Write initialization block to shell configuration
"$MAMBA_EXE" shell init -s "$CURRENT_SHELL" --root-prefix "$MAMBA_ROOT_PREFIX"

# Inject activation hook into current script execution state
eval "$("$MAMBA_EXE" shell hook -s "$CURRENT_SHELL" --root-prefix "$MAMBA_ROOT_PREFIX")"

# Provision environment
"$MAMBA_EXE" create -p "$ENV_PATH" python=3.12 -c conda-forge -c tudat-team tudatpy -y

micromamba clean --all -y

micromamba activate "$ENV_PATH"

# Resolve dependencies and verify binding
uv sync
uv run python -c "import tudatpy; print('Tudatpy Version:', tudatpy.__version__)"

# Cleaning Cache

uv cache clean

# Enforce environment activation on subsequent container attachments
if ! grep -q "micromamba activate" "$RC_FILE"; then
    echo "micromamba activate \"$ENV_PATH\"" >> "$RC_FILE"
fi
