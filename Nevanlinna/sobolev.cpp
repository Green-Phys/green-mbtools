#include <Python.h>
#include "sobolev.h"


/*
GH: Some changes worth trying:
1.  the norm (target function for optimization) is stored as long double
    in the original sobolev.cpp. Here, I am just using double.
2.  We will use Scipy's conjugate gradient code instead of dakota for the time being.
3.  If #2 is too slow, we might have to revert to dakota and Jiani's original implementation.
*/


static PyObject* method_sobolev (PyObject *self, PyObject *args, PyObject *kwargs) {
    // Essential parameters for Hardy optimization
    char *coefile, *param_file, *spectral_file;
    int n_params, n_real;
    double w_min, w_max, eta;

    // Variables with pre-defined default values
    double lagrange = 0.00001;
    
    // Keywords dictionary for input arguments
    static char* kwlist[] = {
        "coefile", "n_params", "n_real", "w_min", "w_max", "eta",
        "param_file", "spectral_file", "lagrange", NULL
    };

    // Parse arguments
    if (
        !PyArg_ParseTupleAndKeywords(
            args, kwargs, "siidddss|d", kwlist,
            &coefile, &n_params, &n_real, &w_min, &w_max, &eta,
            &param_file, &spectral_file, &lagrange
        )
    ) {
        std::cout << "Couldn't parse the input correctly." << std::endl;
        return NULL;
    }

    // Run the sobolev main function with the input parameters
    double norm;
    norm = sobolev_main(
        coefile, n_params, n_real, w_min, w_max, eta,
        param_file, spectral_file, lagrange
    );

    // Optimization
    return PyFloat_FromDouble(norm);
}


static PyMethodDef SobolevMethods[] = {
    {
        "sobolev", (PyCFunction) method_sobolev, METH_VARARGS | METH_KEYWORDS,
        "Python interface for Sobolev norm function for Hardy optimization."
    },
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef sobolevmodule = {
    PyModuleDef_HEAD_INIT,
    "sobolev",
    "Python interface for Sobolev norm function for Hardy optimization.",
    -1,
    SobolevMethods
};


PyMODINIT_FUNC PyInit_sobolev(void) {
    return PyModule_Create(&sobolevmodule);
}
