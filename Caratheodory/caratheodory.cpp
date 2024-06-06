#include <Python.h>
#include <typeinfo>

#include "iter.h"

/**
 * @brief Example function for Caratheodory analytical continuation of matrix-valued imaginary time data.
 * 
 * @param ifile : Input file with data for different Matsubara frequencies
 * @param n_imag : Number of imaginary frequency data points
 * @param dim : Dimension of the matrix, e.g. dim=4 for a 4x4 matrix
 * @param compfile : Output matrix-valued data file
 * @param kresfile : Output spectral function data file
 * @param use_custom_real_grid : whether to use customized real grid
 * @param grid_file : name of real-frequency input file (needed if use_custom_real_grid is 1)
 * @param n_real : number of real frequency points
 * @param w_min : minimum real frequency (needed if use_custom_real_grid is 1)
 * @param w_max : maximum real frequency (needed if use_custom_real_grid is 1)
 * @param eta : (usually small) broadening parameter -- used in the form of w + i eta in continuation
 */
void runCaratheodory (
    std::string ifile, int n_imag, int dim, std::string compfile, std::string kresfile,
    int use_custom_real_grid=false, std::string grid_file="real_grid.txt",
    int n_real=10000, double w_min=-10., double w_max=10., double eta=0.01
) {
    mpfr_set_default_prec(1024);
    real_domain_data<green::ac::mpfr_float> real(
        n_real, dim, use_custom_real_grid, grid_file, w_min, w_max, eta
    );

    // Perform analytic continuation
    Cara<green::ac::mpfr_float> cara(n_imag, dim, ifile);
    std::cout << "Cara class initialized";
    cara.evaluation(real);

    // Write output
    real.compact_write(compfile);
    real.trace_write(kresfile);

    return;
}


static PyObject* method_caratheodory (PyObject *self, PyObject *args, PyObject *kwargs) {
    // Essential parameters to run Caratheodory
    char *ifile, *compfile, *kresfile = NULL;
    int n_imag = 1;
    int dim = 1;

    // Optional parameters with some default values for Caratheodory
    int use_custom_real_grid=0;
    char *grid_file=NULL;
    int n_real = 101;
    double w_min = -5.;
    double w_max = 5.;
    double eta = 0.01;

    // Define keywords dictionary
    static const char* kwlist[] = {
        "ifile", "n_imag", "dim", "compfile", "kresfile", "use_custom_real_grid",
        "grid_file", "n_real", "w_min", "w_max", "eta", NULL
    };

    // Parse arguments
    if (
        !PyArg_ParseTupleAndKeywords(
            args, kwargs, "siiss|isiddd", (char**)kwlist,
            &ifile, &n_imag, &dim, &compfile, &kresfile,
            &use_custom_real_grid, &grid_file, &n_real, &w_min, &w_max, &eta
        )
    ) {
        std::cout << "Couldn't parse the input properly." << std::endl;
        return NULL;
    }

    mpfr_set_default_prec(1024);
    real_domain_data<green::ac::mpfr_float> real(
        n_real, dim, use_custom_real_grid, grid_file, w_min, w_max, eta
    );

    for (int i = 0; i < n_real; i++) {
        std::cout << "freq[" << i << "] = " << real.freq()[i] << std::endl;
    }

    // Perform Caratheodory analytic continuation
    Cara<green::ac::mpfr_float> cara(n_imag, dim, ifile);
    std::cout << "Caratheodory class initialized" << std::endl;
    cara.evaluation(real);

    // Write output
    real.compact_write(compfile);
    real.trace_write(kresfile);

    Py_RETURN_NONE;
}


static PyMethodDef CaratheodoryMethods[] = {
    {"caratheodory", (PyCFunction) method_caratheodory, METH_VARARGS | METH_KEYWORDS, "Python interface for Caratheodory analtical continuation."},
    {NULL, NULL, 0, NULL}
};


static struct PyModuleDef caratheodorymodule = {
    PyModuleDef_HEAD_INIT,
    "caratheodory",
    "Python interface for Caratheodory analytical continuation.",
    -1,
    CaratheodoryMethods
};


PyMODINIT_FUNC PyInit_caratheodory(void) {
    return PyModule_Create(&caratheodorymodule);
}

