C++ Implementation of a Market Model with Kalman-Filtered Estimates of Alpha and Beta Coefficients.

The RHS and LHS variable data should be in the `market.csv` and `stock.csv` files respectively.  When compiling you should link to the `openblas`, `lapack`, `math`, and `nlopt` libraries.  For example, use something like:

> clang++ -Wall kalman.cpp kalman_forward_pass.cpp -o kalman -lopenblas -llapack -lm -lnlopt
