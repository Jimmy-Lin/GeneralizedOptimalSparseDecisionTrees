# This is the code used in the [ICML 2020 paper](https://arxiv.org/abs/2006.08690).
Click [here for for newer code](link to: https://github.com/ubc-systopia/gosdt-guesses) used in the [AAAI 2022 paper](https://www.aaai.org/AAAI22Papers/AAAI-4608.McTavishH.pdf)

# GOSDT Documentation
Implementation of [Generalized Optimal Sparse Decision Tree](https://arxiv.org/pdf/2006.08690.pdf).

# Table of Content
- [Usage](#usage)
- [Development](#development)
- [Configuration](#configuration)
- [Dependencies](#dependencies)
- [License](#license)
- [FAQs](#faqs)

---

# Usage

Guide for end-users who want to use the library without modification.

Describes how to install and use the library as a stand-alone command-line program or as an embedded extension in a larger project.
Currently supported as a Python extension.

## Installing Dependencies
Refer to [**Dependency Installation**](/doc/dependencies.md##Installation)

## As a Stand-Alone Command Line Program
### Installation
```
./autobuild --install
```

### Executing the Program
```bash
gosdt dataset.csv config.json
# or 
cat dataset.csv | gosdt config.json >> output.json
```

For examples of dataset files, refer to `experiments/datasets/compas/binned.csv`.
For an example configuration file, refer to `experiments/configurations/compas.json`.
For documentation on the configuration file, refer to [**Dependency Installation**](/doc/configuration.md)

## As a Python Library with C++ Extensions
### Build and Installation
```
./autobuild --install-python
```
_If you have multiple Python installations, please make sure to build and install using the same Python installation as the one intended for interacting with this library._


### Importing the C++ Extension
```python
import gosdt

with open ("data.csv", "r") as data_file:
    data = data_file.read()

with open ("config.json", "r") as config_file:
    config = config_file.read()


print("Config:", config)
print("Data:", data)

gosdt.configure(config)
result = gosdt.fit(data)

print("Result: ", result)
print("Time (seconds): ", gosdt.time())
print("Iterations: ", gosdt.iterations())
print("Graph Size: ", gosdt.size())
```

### Importing Extension with local Python Wrapper
```python
import pandas as pd
import numpy as np
from model.gosdt import GOSDT

dataframe = pd.DataFrame(pd.read_csv("experiments/datasets/monk_2/data.csv"))

X = dataframe[dataframe.columns[:-1]]
y = dataframe[dataframe.columns[-1:]]

hyperparameters = {
    "regularization": 0.1,
    "time_limit": 3600,
    "verbose": True,
}

model = GOSDT(hyperparameters)
model.fit(X, y)
print("Execution Time: {}".format(model.time))

prediction = model.predict(X)
training_accuracy = model.score(X, y)
print("Training Accuracy: {}".format(training_accuracy))
print(model.tree)
```

---

# Development


Guide for developers who want to use, modify and test the library.

Describes how to install and use the library with details on project structure.

## Repository Structure
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

## Installing Dependencies
Refer to [**Dependency Installation**](/doc/dependencies.md##Installation)

## Build Process
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

---

# Configuration

Details on the configuration options.

```bash
gosdt dataset.csv config.json
# or
cat dataset.csv | gosdt config.json
```

Here the file `config.json` is optional.
There is a default configuration which will be used if no such file is specified.

## Configuration Description

The configuration file is a JSON object and has the following structure and default values:
```json
{
  "balance": false,
  "cancellation": true,
  "look_ahead": true,
  "similar_support": true,
  "feature_exchange": true,
  "continuous_feature_exchange": true,
  "rule_list": false,

  "diagnostics": false,
  "verbose": false,

  "regularization": 0.05,
  "uncertainty_tolerance": 0.0,
  "upperbound": 0.0,

  "model_limit": 1,
  "precision_limit": 0,
  "stack_limit": 0,
  "tile_limit": 0,
  "time_limit": 0,
  "worker_limit": 1,

  "costs": "",
  "model": "",
  "profile": "",
  "timing": "",
  "trace": "",
  "tree": ""
}
```

### Key parameters

**regularization**
 - Values: Decimal within range [0,1]
 - Description: Used to penalize complexity. A complexity penalty is added to the risk in the following way.
   ```
   ComplexityPenalty = # Leaves x regularization
   ```
 - Default: 0.05
 - Note: We highly recommend setting the regularization to a value larger than 1/num_samples. A small regularization could lead to a longer training time. 

**time_limit**
 - Values: Decimal greater than or equal to 0
 - Description: A time limit upon which the algorithm will terminate. If the time limit is reached, the algorithm will terminate with an error.
 - Special Cases: When set to 0, no time limit is imposed.


### Flags

**balance**
 - Values: true or false
 - Description: Enables overriding the sample importance by equalizing the importance of each present class

**cancellation**
 - Values: true or false
 - Description: Enables propagate up the dependency graph of task cancellations

**look_ahead**
 - Values: true or false
 - Description: Enables the one-step look-ahead bound implemented via scopes

**similar_support**
 - Values: true or false
 - Description: Enables the similar support bound imeplemented via the distance index

**feature_exchange**
 - Values: true or false
 - Description: Enables pruning of pairs of features using subset comparison

**continuous_feature_exchange**
 - Values: true or false
 - Description: Enables pruning of pairs continuous of feature thresholds using subset comparison

**diagnostics**
 - Values: true or false
 - Description: Enables printing of diagnostic trace when an error is encountered to standard output

**verbose**
 - Values: true or false
 - Description: Enables printing of configuration, progress, and results to standard output

 ### Tuners

**uncertainty_tolerance**
 - Values: Decimal within range [0,1]
 - Description: Used to allow early termination of the algorithm. Any models produced as a result are guaranteed to score within the lowerbound and upperbound at the time of termination. However, the algorithm does not guarantee that the optimal model is within the produced model unless the uncertainty value has reached 0.

 - Values: Decimal within range [0,1]
 - Description: Used to limit the risk of model search space. This can be used to ensure that no models are produced if even the optimal model exceeds a desired maximum risk. This also accelerates learning if the upperbound is taken from the risk of a nearly optimal model.

### Limits

**model_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of models that will be extracted into the output.
 - Special Cases: When set to 0, no output is produced.

**precision_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of significant figures considered when converting ordinal features into binary features.
 - Special Cases: When set to 0, no limit is imposed.

**stack_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of bytes considered for use when allocating local buffers for worker threads.
 - Special Cases: When set to 0, all local buffers will be allocated from the heap.

**tile_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of bits used for the finding tile-equivalence
 - Special Cases: When set to 0, no tiling is performed.

**worker_limit**
 - Values: Decimal greater than or equal to 1
 - Description: The maximum number of threads allocated to executing th algorithm.
 - Special Cases: When set to 0, a single thread is created for each core detected on the machine.

### Files

**costs**
 - Values: string representing a path to a file.
 - Description: This file must contain a CSV representing the cost matrix for calculating loss.
   - The first row is a header listing every class that is present in the training data
   - Each subsequent row contains the cost incurred of predicitng class **i** when the true class is **j**, where **i** is the row index and **j** is the column index
   - Example where each false negative costs 0.1 and each false positive costs 0.2 (and correct predictions costs 0.0):
     ```
     negative,positive
     0.0,0.1
     0.2,0.0
     ```
   - Example for multi-class objectives:
     ```
     class-A,class-B,class-C
     0.0,0.1,0.3
     0.2,0.0,0.1
     0.8,0.3,0.0
     ```
   - Note: costs values are not normalized, so high cost values lower the relative weight of regularization
 - Special Case: When set to empty string, a default cost matrix is used which represents unweighted training misclassification.

**model**
 - Values: string representing a path to a file.
 - Description: The output models will be written to this file.
 - Special Case: When set to empty string, no model will be stored.

**profile**
 - Values: string representing a path to a file.
 - Description: Various analytics will be logged to this file.
 - Special Case: When set to empty string, no analytics will be stored.

**timing**
 - Values: string representing a path to a file.
 - Description: The training time will be appended to this file.
 - Special Case: When set to empty string, no training time will be stored.

**trace**
 - Values: string representing a path to a directory.
 - Description: snapshots used for trace visualization will be stored in this directory
 - Special Case: When set to empty string, no snapshots are stored.

**tree**
 - Values: string representing a path to a directory.
 - Description: snapshots used for trace-tree visualization will be stored in this directory
 - Special Case: When set to empty string, no snapshots are stored.

## Optimizing Different Loss Functions

When using the Python interface `python/model/gosdt.py` additional loss functions are available.
Here is the list of loss functions implemented along with descriptions of their hyperparameters.

### Accuracy
```
{ "objective": "acc" }
```
This optimizes the loss defined as the uniformly weighted number of misclassifications.

### Balanced Accuracy
```
{ "objective": "bacc" }
```
This optimizes the loss defined as the number of misclassifications, adjusted for imbalanced representation of positive or negative samples.

### Weighted Accuracy
```
{ "objective": "wacc", "w": 0.5 }
```
This optimizes the loss defined as the number of misclassifications, adjusted so that negative samples have a weight of `w` while positive samples have a weight of `1.0`

### F - 1 Score
```
{ "objective": "f1", "w": 0.9 }
```
This optimizes the loss defined as the [F-1](https://en.wikipedia.org/wiki/F1_score) score of the model's predictions.

### Area under the Receiver Operanting Characteristics Curve
```
{ "objective": "auc" }
```
This maximizes the area under the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve formed by varying the prediction of the leaves.

### Partial Area under the Receiver Operanting Characteristics Curve
```
{ "objective": "pauc", "theta": 0.1 }
```
This maximizes the partial area under the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve formed by varying the prediction of the leaves. The area is constrained so that false-positive-rate is in the closed interval `[0,theta]`

---

# Dependencies

List of external dependencies

The following dependencies need to be installed to build the program. 
 - [**Boost**](https://www.boost.org/) - Collection of portable C++ source libraries
 - [**GMP**](http://gmplib.org/) - Collection of functions for high-precision artihmetics
 - [**Intel TBB**](https://www.threadingbuildingblocks.org/) - Rich and complete approach to parallelism in C++
 - [**WiredTiger**](https://source.wiredtiger.com/2.5.2/index.html) - WiredTiger is an high performance, scalable, production quality, NoSQL, Open Source extensible platform for data management
 - [**OpenCL**](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=14&cad=rja&uact=8&ved=2ahUKEwizj4n2k8LlAhVcCTQIHZlADscQFjANegQIAhAB&url=https%3A%2F%2Fwww.khronos.org%2Fregistry%2FOpenCL%2F&usg=AOvVaw3JjOwbrewRqPxpTXRZ6vN9)(Optional) - A framework for execution across heterogeneous hardware accelerators.

### Bundled Dependencies
The following dependencies are included as part of the repository, thus requiring no additional installation.
 - [**nlohmann/json**](https://github.com/nlohmann/json) - JSON Parser
 - [**ben-strasser/fast-cpp-csv-parser**](https://github.com/ben-strasser/fast-cpp-csv-parser) - CSV Parser
 - [**OpenCL C++ Bindings 1.2**](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.2.pdf) - OpenCL bindings for GPU computing

 ### Installation
 Install these using your system package manager.
 There are also installation scripts provided for your convenience: **trainer/auto**
 
 These currently support interface with **brew** and **apt**
  - **Boost** - `auto/boost.sh --install`
  - **GMP** - `auto/gmp.sh --install`
  - **Intel TBB** - `auto/tbb.sh --install`
  - **WiredTiger** - `auto/wiredtiger.sh --install`
  - **OpenCL** - `auto/opencl.sh --install`

---

# FAQs

If you run into any issues, consult the [**FAQs**](/doc/faqs.md) first. 

---

# License

Licensing information

---

**Inquiries**

For general inquiries, send an email to `jimmy.projects.lin@gmail.com`
