#!/bin/sh
if [ "$1" = "--install" ]; then
    echo "Installing Wired Tiger."

    if command -v brew > /dev/null; then
        brew install wiredtiger
    elif command -v apt-get > /dev/null; then
        sudo apt-get install wiredtiger
    else
        echo "Could not detect or recognize system package manager while attempting to install Wired Tiger."
        echo "Please install Wired Tiger on this system and try again."
        exit 1
    fi
elif [ "$1" = "--uninstall" ]; then
    echo "Uninstalling Wired Tiger."

    if command -v brew > /dev/null; then
        brew uninstall wiredtiger
    elif command -v apt-get > /dev/null; then
        sudo apt-get remove wiredtiger
    else
        echo "Could not detect or recognize system package manager while attempting to install Wired Tiger."
        echo "Please install Wired Tiger on this system and try again."
        exit 1
    fi
else
    echo "Please specify action: --install or --uninstall"
fi