#!/bin/sh
if [ "$1" = "--install" ]; then
    echo "Installing OpenCL."

    if command -v brew > /dev/null; then
        echo "OpenCL is bundled with MacOS distributions and therefore is already installed."
    elif command -v apt-get > /dev/null; then
        sudo apt-get install ocl-icd-opencl-dev
    else
        echo "Could not detect or recognize system package manager while attempting to install OpenCL."
        echo "Please install OpenCL on this system and try again."
        exit 1
    fi
elif [ "$1" = "--uninstall" ]; then
    echo "Uninstalling OpenCL."

    if command -v brew > /dev/null; then
        echo "OpenCL is bundled with MacOS distributions and therefore cannot be removed."
    elif command -v apt-get > /dev/null; then
        sudo apy-get remove ocl-icd-opencl-dev
    else
        echo "Could not detect or recognize system package manager while attempting to install OpenCL."
        echo "Please install OpenCL on this system and try again."
        exit 1
    fi
else
    echo "Please specify action: --install or --uninstall"
fi