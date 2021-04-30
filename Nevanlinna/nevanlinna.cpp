#include "schur.h"


int main (int argc, char * argv[]) {
    std::string ifile, ofile, coefile;
    int imag_num;
    //prompt user for input parameters
    std::cin >> ifile >> imag_num >> ofile >> coefile;
    //set calculation precision
    mpf_set_default_prec(1024);
    //begin evaluation
    Schur<mpf_class> NG(ifile, imag_num, ofile);
    NG.evaluation(coefile);
    return 0;
}