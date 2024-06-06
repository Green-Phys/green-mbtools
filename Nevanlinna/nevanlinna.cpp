#include <Python.h>
#include <typeinfo>
#include "schur.h"

void runNevanlinna (
    std::string ifile, int n_imag, std::string ofile, std::string coefile, int prec=128,
    bool spectral=false, int n_real=10000, double w_min=-10., double w_max=10., double eta=0.01
) {
    mpf_set_default_prec(prec);
    Schur<green::ac::gmp_float> NG(ifile, n_imag, ofile, n_real, w_min, w_max, eta);
    std::cout << "Schur class initialized";
    NG.evaluation(coefile, spectral);

    return;
}


static PyObject* method_nevanlinna (PyObject *self, PyObject *args, PyObject *kwargs) {
    //Essential parameters to run Nevanlinna
    char *ifile, *ofile, *coefile = NULL;
    int n_imag = 1;

    //Optional parameters with some default values for Nevanlinna
    int prec = 128;
    int spectral = 0;
    int n_real = 10000;
    double w_min = -15.;
    double w_max = 15.;
    double eta = 0.01;

    //Define keywords dictionary
    static const char* kwlist[] = {
        "ifile", "n_imag", "ofile", "coefile", "prec", "spectral", "n_real", "w_min", "w_max", "eta", NULL
    };

    // Parse arguments
    if (
        !PyArg_ParseTupleAndKeywords(
            args, kwargs, "siss|iiiddd", (char**)kwlist,
            &ifile, &n_imag, &ofile, &coefile, &prec, &spectral, &n_real, &w_min, &w_max, &eta
        )
    ) {
        std::cout << "Couldn't parse the input properly." << std::endl;
        return NULL;
    }

    // Perform the Nevanlinna analytic continuation for the given files and parameters
    mpf_set_default_prec(prec);
    Schur<green::ac::gmp_float> NG(ifile, n_imag, ofile, n_real, w_min, w_max, eta);
    NG.evaluation(coefile, spectral);

    Py_RETURN_NONE;
}


static PyMethodDef NevanlinnaMethods[] = {
    {"nevanlinna", (PyCFunction) method_nevanlinna, METH_VARARGS | METH_KEYWORDS, "Python interface for nevanlinna analtical continuation."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef nevanlinnamodule = {
    PyModuleDef_HEAD_INIT,
    "nevanlinna",
    "Python interface for nevanlinna analytical continuation.",
    -1,
    NevanlinnaMethods
};


PyMODINIT_FUNC PyInit_nevanlinna(void) {
    return PyModule_Create(&nevanlinnamodule);
}
