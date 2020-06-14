#!/bin/sh
if [ "$1" = "--install" ]; then
    echo "Installing Intel TBB."

    if command -v brew > /dev/null; then
        brew install tbb
    elif command -v apt-get > /dev/null; then
        sudo apt-get install libtbb-dev
    else
        echo "Could not detect or recognize system package manager while attempting to install Intel TBB."
        echo "Please install Intel TBB on this system and try again."
        exit 1
    fi
elif [ "$1" = "--uninstall" ]; then
    echo "Uninstalling Intel TBB."

    if command -v brew > /dev/null; then
        brew uninstall tbb
    elif command -v apt-get > /dev/null; then
        sudo apt-get remove libtbb-dev
    else
        echo "Could not detect or recognize system package manager while attempting to install Intel TBB."
        echo "Please install Intel TBB on this system and try again."
        exit 1
    fi
else
    echo "Please specify action: --install or --uninstall"
fi