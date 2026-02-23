#!/bin/bash
# Creates a symbolic link to this directory in the Syntalos modules directory.
# This allows Syntalos to find the module while being able to modify the module
# code without reinstalling it.

if [ "$(uname)" != "Linux" ]; then
    echo "Script intended for Linux!"
    exit 1
fi

if [ ! -f "module.toml" ]; then
    echo "Run this script from the module directory!"
    exit 1
fi

PWD=$(pwd)
MODULE_SRC="$PWD/"
MODULE_DIR_NAME=$(basename $PWD | sed 's/syntalos_//')

# Check if flatpak is installed and if Syntalos is installed and choose the correct path
if command -v flatpak &> /dev/null && flatpak list | grep -q org.syntalos.syntalos; then
    SYNTALOS_MODULES_DIR="$HOME/.var/app/org.syntalos.syntalos/data/modules"
else
    SYNTALOS_MODULES_DIR="$HOME/.local/share/Syntalos/modules"
fi

mkdir -p $SYNTALOS_MODULES_DIR
SYMLINK="$SYNTALOS_MODULES_DIR/$MODULE_DIR_NAME"

# The syntax of `ln` is ambiguous when the target already exists.
# If we don't remove the existing symlink, `ln` will create a symlink in the
# current directory
if [ -L "$SYMLINK" ]; then
    echo "Symlink already exists, removing it..."
    rm "$SYMLINK"
fi

ln --verbose -s $MODULE_SRC $SYMLINK