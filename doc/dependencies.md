[Back](/README.md)

## Dependencies
The following dependencies need to be installed to build the program. 
 - [**Boost**](https://www.boost.org/) - Collection of portable C++ source libraries
 - [**OpenCL**](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=14&cad=rja&uact=8&ved=2ahUKEwizj4n2k8LlAhVcCTQIHZlADscQFjANegQIAhAB&url=https%3A%2F%2Fwww.khronos.org%2Fregistry%2FOpenCL%2F&usg=AOvVaw3JjOwbrewRqPxpTXRZ6vN9) - A framework for execution across heterogeneous hardware accelerators.
 - [**Intel TBBs**](https://www.threadingbuildingblocks.org/) - Rich and complete approach to parallelism in C++

## Bundled Dependencies
The following dependencies are included as part of the repository, thus requireing no additional installation.
 - [**nlohmann/json**](https://github.com/nlohmann/json) - JSON Parser
 - [**ben-strasser/fast-cpp-csv-parser**](https://github.com/ben-strasser/fast-cpp-csv-parser) - CSV Parser
 - [**OpenCL C++ Bindings 1.2**](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.2.pdf) - OpenCL bindings for GPU computing

 ## Installation
 Install these using your system package manager.
 There are also installation scripts provided for your convenience: **trainer/auto**
 
 These currently support interface with **brew** and **apt**
  - **Boost** - `trainer/auto/boost.sh --install`
  - **OpenCL** - `trainer/auto/opencl.sh --install`
  - **Intel TBB** - `trainer/auto/tbb.sh --install`