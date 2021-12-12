#include <iostream>
#include "kalman.h"

#include "armadillo"

using namespace arma;
using namespace std;

using arma::colvec;
using arma::log;
using arma::normcdf;
using std::exp;
using std::log;
using std::pow;
using std::sqrt;

#define PII 3.1415926

int forward_pass(double theta[3])
{

  colvec stock;
  stock.load("./stock.csv", csv_ascii);

  colvec market;
  market.load("./market.csv", csv_ascii);
  
  arma::vec y = stock;
  mat X(y.n_elem, 1, fill::ones);
  X.insert_cols(1, market);
 
  double alpha0_hat = theta[0];
  double alpha1_hat = theta[1];
  double alpha2_hat = theta[2];

  int num_observations = y.n_elem;
  int num_variables = X.n_cols;

  arma::mat F;
  F.eye(num_variables, num_variables);

  double R = pow(alpha0_hat, 2);

  // Q is a kxk matrix of parameters we must estimate:  I have made it diagonal.
  // use the below when we improve
  // arma::mat Q = arma::mat(num_variables, num_variables, fill::zeros);
  // with the translated R diag(Q) <- c(theta[2]^2, theta[3]^2)
  
  // TODO: need to adjust this manually at this point
  
  arma::mat Q;
  Q << alpha1_hat << 0 << arma::endr
    << 0 << alpha2_hat << arma::endr;

  //pick up kalman filter here with betatt

  arma::mat betatt = arma::mat(num_observations, num_variables, arma::fill::zeros);
  arma::mat betatt_1 = arma::mat(num_observations, num_variables, arma::fill::zeros);
  
  arma::vec eta(num_observations, arma::fill::zeros);
  arma::vec f(num_observations, arma::fill::zeros);

  arma::mat Ptt = arma::mat(num_observations, pow(num_variables, 2), arma::fill::zeros);
  arma::mat Ptt_1 = arma::mat(num_observations, pow(num_variables, 2), arma::fill::zeros);

  // # need to set initial values for beta and p
  // # make Ptt based on some estimate

  arma::vec first_row_betatt = (arma::inv(X.t() * X) * X.t()) * y;
  arma::vec first_row_Ptt = arma::vec(pow(num_variables, 2));
  first_row_Ptt.fill(.3);

  betatt.row(0) = first_row_betatt.t();
  Ptt.row(0) = first_row_Ptt.t();

  for (int i = 1; i < num_observations; i++){

    ///Prediction
    betatt_1.row(i) = (F * betatt.row(i - 1).t()).t();
    //ok so the next line converts (on the fly) the ith row of Ptt to a matrix.  Would be better to use an array...
    arma::mat tempPtt(2, 2);
    tempPtt << Ptt.row(i - 1)[0] << Ptt.row(i - 1)[2] << arma::endr
	    << Ptt.row(i - 1)[1] << Ptt.row(i - 1)[3] << arma::endr;
      
    Ptt_1.row(i) = arma::vectorise(F * tempPtt * F.t() + Q).t();

    eta.at(i) = y.at(i) - dot(X.row(i), betatt_1.row(i));

      //temp matrix
      arma::mat tempPtt_1(2, 2);
          tempPtt_1 << Ptt_1.row(i)[0] << Ptt_1.row(i)[2] << arma::endr
	    << Ptt_1.row(i)[1] << Ptt_1.row(i)[3] << arma::endr;

	  f.at(i) = as_scalar(X.row(i) * tempPtt_1 * X.row(i).t()) + R;

    /// Updating

	  arma::mat as_mat_tempPtt_1(2, 2);
	  as_mat_tempPtt_1 << Ptt_1.row(i)[0] << Ptt_1.row(i)[2] << arma::endr
			   << Ptt_1.row(i)[1] << Ptt_1.row(i)[3] << arma::endr;

	  betatt.row(i) = betatt_1.row(i) + (((as_mat_tempPtt_1 * X.row(i).t()) * eta.at(i)) * (1 / f.at(i))).t();

	  Ptt.row(i) = arma::vectorise(as_mat_tempPtt_1 - as_mat_tempPtt_1 * X.row(i).t() * X.row(i) * as_mat_tempPtt_1 * (1 / f.at(i))).t();

  }

  //the first element of f is 0, so change to 1---log(1) is 0 and 1/1 is 1, so OK
  //f.at(0) = 1;

  //  double logl = -0.5 * arma::accu(log(abs(f))) - 0.5 * arma::accu(eta % eta * (1 / f).t());

  betatt.col(1).print();

  //forecast next stock return value and standard deviation-----------

  //arma::vec prediction =

  std::cout << "\nPrediction:\n" << X.row(num_observations - 1) * betatt.row(num_observations - 1).t() << std::endl;

  std::cout << "\nStandard deviation of previous prediction error:\n" << sqrt(f.at(num_observations - 1)) << std::endl;

  return 0;

}
