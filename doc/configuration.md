[Back](/README.md)

# Configuration

Training the decision tree can be customized by passing in an optional path to a configuration file as so:

```bash
gosdt dataset.csv config.json
# or
cat dataset.csv | gosdt config.json
```

Here the file `config.json` is optional.
There is a default configuration which will be used if no such file is specified.

# Configuration Description

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
  "non_binary": false,

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

## Flags

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

**verbose**
 - Values: true or false
 - Description: Enables non-binary encoding (only supported in command line)
 ## Tuners

 **regularization**
 - Values: Decimal within range [0,1]
 - Description: Used to penalize complexity. A complexity penalty is added to the risk in the following way.
   ```
   ComplexityPenalty = # Leaves x regularization
   ```

 **uncertainty_tolerance**
 - Values: Decimal within range [0,1]
 - Description: Used to allow early termination of the algorithm. Any models produced as a result are guaranteed to score within the lowerbound and upperbound at the time of termination. However, the algorithm does not guarantee that the optimal model is within the produced model unless the uncertainty value has reached 0.

 - Values: Decimal within range [0,1]
 - Description: Used to limit the risk of model search space. This can be used to ensure that no models are produced if even the optimal model exceeds a desired maximum risk. This also accelerates learning if the upperbound is taken from the risk of a nearly optimal model.

## Limits

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

**time_limit**
 - Values: Decimal greater than or equal to 0
 - Description: A time limit upon which the algorithm will terminate. If the time limit is reached, the algorithm will terminate with an error.
 - Special Cases: When set to 0, no time limit is imposed.

**worker_limit**
 - Values: Decimal greater than or equal to 1
 - Description: The maximum number of threads allocated to executing th algorithm.
 - Special Cases: When set to 0, a single thread is created for each core detected on the machine.

## Files

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

# Optimizing Different Loss Functions

When using the Python interface `python/model/gosdt.py` additional loss functions are available.
Here is the list of loss functions implemented along with descriptions of their hyperparameters.

## Accuracy
```
{ "objective": "acc" }
```
This optimizes the loss defined as the uniformly weighted number of misclassifications.

## Balanced Accuracy
```
{ "objective": "bacc" }
```
This optimizes the loss defined as the number of misclassifications, adjusted for imbalanced representation of positive or negative samples.

## Weighted Accuracy
```
{ "objective": "wacc", "w": 0.5 }
```
This optimizes the loss defined as the number of misclassifications, adjusted so that negative samples have a weight of `w` while positive samples have a weight of `1.0`

## F - 1 Score
```
{ "objective": "f1" }
```
This optimizes the loss defined as the [F-1](https://en.wikipedia.org/wiki/F1_score) score of the model's predictions.

## Area under the Receiver Operanting Characteristics Curve
```
{ "objective": "auc" }
```
This maximizes the area under the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve formed by varying the prediction of the leaves.

## Partial Area under the Receiver Operanting Characteristics Curve
```
{ "objective": "pauc", "theta": 0.1 }
```
This maximizes the partial area under the [ROC](https://en.wikipedia.org/wiki/Receiver_operating_characteristic) curve formed by varying the prediction of the leaves. The area is constrained so that false-positive-rate is in the closed interval `[0,theta]`