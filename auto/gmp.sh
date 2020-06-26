#!/bin/sh
if [ "$1" = "--install" ]; then
    echo "Installing GMP."

    if command -v brew > /dev/null; then
        brew install gmp
    elif command -v apt-get > /dev/null; then
        sudo apt-get install libgmp-dev
        sudo apt-get install libgmp10
    else
        echo "Could not detect or recognize system package manager while attempting to install GMP."
        echo "Please install GMP on this system and try again."
        exit 1
    fi
elif [ "$1" = "--uninstall" ]; then
    echo "Uninstalling GMP."

    if command -v brew > /dev/null; then
        brew uninstall gmp
    elif command -v apt-get > /dev/null; then
        sudo apt-get remove libbmp-dev
        sudo apt-get remove libgmp10
    else
        echo "Could not detect or recognize system package manager while attempting to install GMP."
        echo "Please install GMP on this system and try again."
        exit 1
    fi
else
    echo "Please specify action: --install or --uninstall"
fi