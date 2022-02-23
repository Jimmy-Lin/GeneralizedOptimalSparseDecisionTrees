if command -v brew > /dev/null; then
        brew install tbb@2020
        brew unlink tbb
        brew link tbb@2020
    elif command -v apt-get > /dev/null; then
        sudo apt-get -y install libtbb-dev
    else
        echo "Could not detect or recognize system package manager while attempting to install Intel TBB."
        echo "Please install Intel TBB 2020_U3 or earlier on this system and try again."
fi
if command -v brew > /dev/null; then
        brew install gmp
    elif command -v apt-get > /dev/null; then
        sudo apt-get -y install libgmp-dev
    else
        echo "Could not detect or recognize system package manager while attempting to install GMP."
        echo "Please install GMP on this system and try again."

fi
if command -v brew > /dev/null; then
      brew install boost
  elif command -v apt-get > /dev/null; then
      sudo apt-get -y install libboost-all-dev
  else
      echo "Could not detect or recognize system package manager while attempting to install Boost."
      echo "Please install Boost on this system and try again."
fi