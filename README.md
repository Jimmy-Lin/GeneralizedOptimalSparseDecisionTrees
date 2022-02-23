# Fast Sparse Decision Tree Optimization via Reference Ensembles

# Table of Content
- [Installation](#installation)
- [Configuration](#configuration)
- [Example](#example)
- [Repository Structure](#repository_structure)
- [License](#license)
- [FAQs](#faqs)

---

# Installation

<!--Guide for end-users who want to use the library without modification.

Describes how to install and use the library as a stand-alone command-line program or as an embedded extension in a larger project.
Currently supported as a Python extension.
-->
### Installing GOSDT Dependencies

List of external dependencies

The following dependencies need to be installed to build the program. 
 - [**Boost**](https://www.boost.org/) - Collection of portable C++ source libraries
 - [**GMP**](http://gmplib.org/) - Collection of functions for high-precision artihmetics
 - [**Intel TBB**](https://www.threadingbuildingblocks.org/) - Rich and complete approach to parallelism in C++
 - [**WiredTiger**](https://source.wiredtiger.com/2.5.2/index.html) - WiredTiger is an high performance, scalable, production quality, NoSQL, Open Source extensible platform for data management
 - [**OpenCL**](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=14&cad=rja&uact=8&ved=2ahUKEwizj4n2k8LlAhVcCTQIHZlADscQFjANegQIAhAB&url=https%3A%2F%2Fwww.khronos.org%2Fregistry%2FOpenCL%2F&usg=AOvVaw3JjOwbrewRqPxpTXRZ6vN9)(Optional) - A framework for execution across heterogeneous hardware accelerators.

**Bundled Dependencies**

The following dependencies are included as part of the repository, thus requiring no additional installation.
 - [**nlohmann/json**](https://github.com/nlohmann/json) - JSON Parser
 - [**ben-strasser/fast-cpp-csv-parser**](https://github.com/ben-strasser/fast-cpp-csv-parser) - CSV Parser
 - [**OpenCL C++ Bindings 1.2**](https://www.khronos.org/registry/OpenCL/specs/opencl-cplusplus-1.2.pdf) - OpenCL bindings for GPU computing

 **Installation**
 
 Install these using your system package manager.
 There are also installation scripts provided for your convenience: **trainer/auto**
 
 These currently support interface with **brew** and **apt**
  - **Boost** - `auto/boost.sh --install`
  - **GMP** - `auto/gmp.sh --install`
  - **Intel TBB** - `auto/tbb.sh --install`
  - **WiredTiger** - `auto/wiredtiger.sh --install`
  - **OpenCL** - `auto/opencl.sh --install`


### Installing GOSDT as a Stand-Alone Command Line Program
```
./autobuild --install
```

### Installing GOSDT as a Python Library with C++ Extensions
```
./autobuild --install-python
```
For more details about GOSDT installation and examples of running **pure** GOSDT, please see https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees/blob/master/README.md for more details.


---

# Configuration

The configuration is a JSON object and has the following structure and default values:
```json
{ 
  "regularization": 0.05,
  "depth_budget": 0,
  "reference_LB": false, 
  "path_to_labels": "",
  "time_limit": 0,
  
  "uncertainty_tolerance": 0.0,
  "upperbound": 0.0
  "worker_limit": 1,
  "stack_limit": 0,
  "precision_limit": 0,
  "model_limit": 1,
  "verbose": false,
  "diagnostics": false,
  "balance": false,
  "look_ahead": true,
  "similar_support": true,
  "cancellation": true,
  "continuous_feature_exchange": false,
  "feature_exchange": false,
  "feature_transform": true,
  "rule_list": false,
  "non_binary": false,
  "costs": "",
  "model": "",
  "timing": "",
  "trace": "",
  "tree": "",
  "profile": ""
}
```

## Key parameters

**regularization**
 - Values: Decimal within range [0,1]
 - Description: Used to penalize complexity. A complexity penalty is added to the risk in the following way.
   ```
   ComplexityPenalty = # Leaves x regularization
   ```
 - Default: 0.05

**depth_budget**
- Values: Integers >= 1
- Description: Used to set the maximum tree depth for solutions, counting a tree with just the root node as depth 1. 0 means unlimited.
- Default: 0

**reference_LB**
 - Values: true or false
 - Description: Enables using a vector of misclassifications from another (reference) model to lower bound our own misclassifications
 - Default: false

**path_to_labels**
- Values: String representing a path to a file. 
- Description: This file must be a single-column csv representing a class prediction for each training observation (in the same order as for the training data, using the same class labels as for the training data, and predicting each class present in the training set at least once across all training points). Typically this csv is obtained by fitting a [gradient boosted decision tree](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingClassifier.html) model on the training data, and saving its training set predictions as a csv file. 
- Example for a dataset with classes 1 and 0: 
     ```
     predicted_class
     0
     1
     1
     1
     0
     ```
- Default: Emptry string

**time_limit**
 - Values: Decimal greater than or equal to 0
 - Description: A time limit upon which the algorithm will terminate. If the time limit is reached, the algorithm will terminate with an error.
 - Special Cases: When set to 0, no time limit is imposed.
 - Default: 0


## More parameters
### Flag

**balance**
 - Values: true or false
 - Description: Enables overriding the sample importance by equalizing the importance of each present class
 - Default: false

**cancellation**
 - Values: true or false
 - Description: Enables propagate up the dependency graph of task cancellations
 - Default: true

**look_ahead**
 - Values: true or false
 - Description: Enables the one-step look-ahead bound implemented via scopes
 - Default: true

**similar_support**
 - Values: true or false
 - Description: Enables the similar support bound imeplemented via the distance index
 - Default: true

**feature_exchange**
 - Values: true or false
 - Description: Enables pruning of pairs of features using subset comparison
 - Default: false

**continuous_feature_exchange**
 - Values: true or false
 - Description: Enables pruning of pairs continuous of feature thresholds using subset comparison
 - Default: false

**feature_transform**
 - Values: true or false
 - Description: Enables the equivalence discovery through simple feature transformations
 - Default: true

**rule_list**
 - Values: true or false
 - Description: Enables rule-list constraints on models
 - Default: false
 
**non_binary**
 - Values: true or false
 - Description: Enables non-binary encoding
 - Default: false

**diagnostics**
 - Values: true or false
 - Description: Enables printing of diagnostic trace when an error is encountered to standard output
 - Default: false

**verbose**
 - Values: true or false
 - Description: Enables printing of configuration, progress, and results to standard output
 - Default: false




### Tuners

**uncertainty_tolerance**
 - Values: Decimal within range [0,1]
 - Description: Used to allow early termination of the algorithm. Any models produced as a result are guaranteed to score within the lowerbound and upperbound at the time of termination. However, the algorithm does not guarantee that the optimal model is within the produced model unless the uncertainty value has reached 0.
 - Default: 0.0

**upperbound**
 - Values: Decimal within range [0,1]
 - Description: Used to limit the risk of model search space. This can be used to ensure that no models are produced if even the optimal model exceeds a desired maximum risk. This also accelerates learning if the upperbound is taken from the risk of a nearly optimal model.
 - Special Cases: When set to 0, the bound is not activated. 
 - Default: 0.0

### Limits
 
**model_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of models that will be extracted into the output.
 - Special Cases: When set to 0, no output is produced.
 - Default: 1

**precision_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of significant figures considered when converting ordinal features into binary features.
 - Special Cases: When set to 0, no limit is imposed.
 - Default: 0

**stack_limit**
 - Values: Decimal greater than or equal to 0
 - Description: The maximum number of bytes considered for use when allocating local buffers for worker threads.
 - Special Cases: When set to 0, all local buffers will be allocated from the heap.
 - Default: 0


**worker_limit**
 - Values: Decimal greater than or equal to 1
 - Description: The maximum number of threads allocated to executing th algorithm.
 - Special Cases: When set to 0, a single thread is created for each core detected on the machine.
 - Default: 1

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
 - Default: Emptry string

**model**
 - Values: string representing a path to a file.
 - Description: The output models will be written to this file.
 - Special Case: When set to empty string, no model will be stored.
 - Default: Emptry string

**profile**
 - Values: string representing a path to a file.
 - Description: Various analytics will be logged to this file.
 - Special Case: When set to empty string, no analytics will be stored.
 - Default: Emptry string

**timing**
 - Values: string representing a path to a file.
 - Description: The training time will be appended to this file.
 - Special Case: When set to empty string, no training time will be stored.
 - Default: Emptry string

**trace**
 - Values: string representing a path to a directory.
 - Description: snapshots used for trace visualization will be stored in this directory
 - Special Case: When set to empty string, no snapshots are stored.
 - Default: Emptry string

**tree**
 - Values: string representing a path to a directory.
 - Description: snapshots used for trace-tree visualization will be stored in this directory
 - Special Case: When set to empty string, no snapshots are stored.
 - Default: Emptry string

---
# Example

Example code to run GOSDT with threshold guessing, lower bound guessing, and depth limit. The example python file is avaialable in python/example.py

```
import pandas as pd
import numpy as np
import time
import pathlib
from sklearn.ensemble import GradientBoostingClassifier
from model.threshold_guess import compute_thresholds
from model.gosdt import GOSDT

# read the dataset
df = pd.read_csv("experiments/datasets/fico.csv", sep=";")
X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
h = df.columns[:-1]

# GBDT parameters for threshold and lower bound guesses
n_est = 40
max_depth = 1

# guess thresholds
X = pd.DataFrame(X, columns=h)
print("X:", X.shape)
print("y:",y.shape)
X_train, thresholds, header, threshold_guess_time = compute_thresholds(X, y, n_est, max_depth)
y_train = pd.DataFrame(y)

# guess lower bound
start_time = time.perf_counter()
clf = GradientBoostingClassifier(n_estimators=n_est, max_depth=max_depth, random_state=42)
clf.fit(X_train, y_train.values.flatten())
warm_labels = clf.predict(X_train)
elapsed_time = time.perf_counter() - start_time
lb_time = elapsed_time

# save the labels from lower bound guesses as a tmp file and return the path to it.
labelsdir = pathlib.Path('/tmp/warm_lb_labels')
labelsdir.mkdir(exist_ok=True, parents=True)
labelpath = labelsdir / 'warm_label.tmp'
labelpath = str(labelpath)
pd.DataFrame(warm_labels).to_csv(labelpath, header="class_labels",index=None) # TODO: verify this formats correctly for gosdt (shouldn't require headers)


# train GOSDT model
config = {
            "regularization": 0.001,
            "depth_budget": 5,
            "warm_LB": True,
            "path_to_labels": labelpath,
            "time_limit": 1800,
            "similar_support": False
        }

model = GOSDT(config)

model.fit(X_train, y_train)

print("evaluate the model, extracting tree and scores", flush=True)

# get the results
train_acc = model.score(X_train, y_train)
n_leaves = model.leaves()
n_nodes = model.nodes()
time = model.utime

print("Model training time: {}".format(time))
print("Training accuracy: {}".format(train_acc))
print("# of leaves: {}".format(n_leaves))
print(model.tree)
```

**Output**

```
X: (10459, 23)
y: (10459,)
gosdt reported successful execution
training completed. 1.658/0.098/1.756 (user, system, wall), mem=364 MB
bounds: [0.290914..0.290914] (0.000000) loss=0.282914, iterations=13569
evaluate the model, extracting tree and scores
Model training time: 1.6584229469299316
Training accuracy: 0.7170857634573095
# of leaves: 8
if ExternalRiskEstimate<=67.5 = 1 and MSinceMostRecentInqexcl7days<=-7.5 = 1 then:
    predicted class: 1
    misclassification penalty: 0.027
    complexity penalty: 0.001

else if ExternalRiskEstimate<=67.5 != 1 and MSinceMostRecentInqexcl7days<=-7.5 = 1 then:
    predicted class: 0
    misclassification penalty: 0.006
    complexity penalty: 0.001

else if ExternalRiskEstimate<=74.5 = 1 and MSinceMostRecentInqexcl7days<=-7.5 != 1 and MSinceMostRecentInqexcl7days<=0.5 = 1 and PercentTradesWBalance<=80.5 = 1 then:
    predicted class: 1
    misclassification penalty: 0.071
    complexity penalty: 0.001

else if ExternalRiskEstimate<=74.5 != 1 and MSinceMostRecentInqexcl7days<=-7.5 != 1 and MSinceMostRecentInqexcl7days<=0.5 = 1 and PercentTradesWBalance<=80.5 = 1 then:
    predicted class: 0
    misclassification penalty: 0.061
    complexity penalty: 0.001

else if ExternalRiskEstimate<=78.5 = 1 and MSinceMostRecentInqexcl7days<=-7.5 != 1 and MSinceMostRecentInqexcl7days<=0.5 = 1 and PercentTradesWBalance<=80.5 != 1 then:
    predicted class: 1
    misclassification penalty: 0.033
    complexity penalty: 0.001

else if ExternalRiskEstimate<=78.5 != 1 and MSinceMostRecentInqexcl7days<=-7.5 != 1 and MSinceMostRecentInqexcl7days<=0.5 = 1 and PercentTradesWBalance<=80.5 != 1 then:
    predicted class: 0
    misclassification penalty: 0.005
    complexity penalty: 0.001

else if ExternalRiskEstimate<=67.5 = 1 and MSinceMostRecentInqexcl7days<=-7.5 != 1 and MSinceMostRecentInqexcl7days<=0.5 != 1 then:
    predicted class: 1
    misclassification penalty: 0.026
    complexity penalty: 0.001

else if ExternalRiskEstimate<=67.5 != 1 and MSinceMostRecentInqexcl7days<=-7.5 != 1 and MSinceMostRecentInqexcl7days<=0.5 != 1 then:
    predicted class: 0
    misclassification penalty: 0.054
    complexity penalty: 0.001
```

---

# Repository_Structure
Guide for developers who want to use, modify and test the library.

Describes how to install and use the library with details on project structure.

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

---

# FAQs

If you run into any issues when running GOSDT, consult the [**FAQs**](https://github.com/Jimmy-Lin/GeneralizedOptimalSparseDecisionTrees/blob/master/doc/faqs.md) first. 

---

# License

This software is licensed under a 3-clause BSD license (see the LICENSE file for details). 

---

