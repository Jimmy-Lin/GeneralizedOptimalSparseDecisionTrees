[Back](/README.md)

# Development
For developers.

Describes the project structure and tooling available.

# Repository Structure
 - **notebooks** - interactive notebooks for examples and visualizations
 - **experiments** - configurations, datasets, and models to run experiments
 - **doc** - documentation
 - **python** - code relating to the Python implementation and wrappers around C++ implementation
 - **auto** - automations for checking and installing project dependencies
 - **dist** - compiled binaries for distribution
 - **build** - compiled binary objects and other build artifacts
 - **lib** - headers for external libraries
 - **log** - log files
 - **src** - source files
 - **test** - test files

# Installing Dependencies
Refer to [**Dependency Installation**](/doc/dependencies.md##Installation)

# Build Process
 - **Check Updates to the Dependency Tests or Makefile** 
   ```
   ./autobuild --regenerate
   ```
 - **Check for Missing Dependencies** 
   ```
   ./autobuild --configure --enable-tests
   ```
 - **Build and Run Test Suite**
   ```
   ./autobuild --test
   ```
 - **Build and Install Program**
   ```
   ./autobuild --install --enable-tests
   ```
 - **Run the Program** 
   ```
   gosdt dataset.csv config.json
   ```
 - **Build and Install the Python Extension**
   ```
   ./autobuild --install-python
   ```
 For a full list of build options, run `./autobuild --help`
