[Back](/README.md)

# Usage
For users.

Describes how to use the library as a stand-alone program or part of a larger application.

# Installing Dependencies
Refer to [**Dependency Installation**](/doc/dependencies.md##Installation)

# As a Stand-Alone Command Line Program
## Installation
```
./autobuild --install
```

## Executing the Program
```bash
gosdt dataset.csv config.json
# or 
cat dataset.csv | gosdt config.json >> output.json
```

For examples of dataset files, refer to `experiments/datasets/compas/binned.csv`.
For an example configuration file, refer to `experiments/configurations/compas.json`.
For documentation on the configuration file, refer to [**Dependency Installation**](/doc/configuration.md)

# As a Python Library with C++ Extensions
## Build and Installation
```
./autobuild --install-python
```
_If you have multiple Python installations, please make sure to build and install using the same Python installation as the one intended for interacting with this library._

---
## Importing the C++ Extension
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

## Importing Extension with local Python Wrapper
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
