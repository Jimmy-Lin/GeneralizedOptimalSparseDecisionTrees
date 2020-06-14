#include "python_extension.hpp"

static PyObject *fit(PyObject *self, PyObject *args) {

    const char * dataset;
    const char * configuration;

    if (!PyArg_ParseTuple(args, "ss", &dataset, &configuration)) {
        return NULL;
    }

    std::istringstream data_stream(dataset);
    std::istringstream config_stream(configuration);
    GOSDT model = GOSDT(config_stream);
    std::string result = model.fit(data_stream);

    return Py_BuildValue("s", result.c_str());
}

// Define the list of methods for a module
static PyMethodDef gosdt_methods[] = {
    // { method name, method pointer, method parameter format, method description }
    {"fit", fit, METH_VARARGS, "Trains the model"}, 
    {NULL, NULL, 0, NULL}
};

// Define the module
static struct PyModuleDef gosdt = {
    PyModuleDef_HEAD_INIT,
    "gosdt", // Module Name
    "Generalized Optimal Sparse Decision Tree", // Module Description
    -1, // Size of per-interpreter state
    gosdt_methods // Module methods
};

// Initialize the module
PyMODINIT_FUNC PyInit_gosdt(void) {
    return PyModule_Create(&gosdt);
}