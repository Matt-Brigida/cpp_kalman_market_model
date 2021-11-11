#include <iostream>

#include "armadillo"

using namespace arma;
using namespace std;

int main(int argc, char** argv)
  {
    //import market and stock--------
    mat stock;
    stock.load("./stock.csv", csv_ascii);

    mat market;
    market.load("./market.csv", csv_ascii);



    //can use function minimization from the GSL here: https://www.gnu.org/software/gsl/doc/html/multimin.html#

  }
