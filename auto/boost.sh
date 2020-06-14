#!/bin/sh
if [ "$1" = "--install" ]; then
    echo "Installing Boost."

    if command -v brew > /dev/null; then
        brew install boost
    elif command -v apt-get > /dev/null; then
        sudo apt-get install liboost-dev
        sudo apt-get install liboost-all-dev
    else
        echo "Could not detect or recognize system package manager while attempting to install Boost."
        echo "Please install Boost on this system and try again."
        exit 1
    fi
elif [ "$1" = "--uninstall" ]; then
    echo "Uninstalling Boost."

    if command -v brew > /dev/null; then
        brew uninstall boost
    elif command -v apt-get > /dev/null; then
        sudo apt-get remove liboost-dev
        sudo apt-get remove liboost-all-dev
    else
        echo "Could not detect or recognize system package manager while attempting to install Boost."
        echo "Please install Boost on this system and try again."
        exit 1
    fi
else
    echo "Please specify action: --install or --uninstall"
fi