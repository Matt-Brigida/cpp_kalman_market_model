#include <iostream>
#include <nlopt.h>

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

// Start Likelihood -------------
// double lik(const double *theta0 ,const double *theta1, const double *theta2, vec y_, vec X_) { //removed vec theta
double lik(double theta[3], vec y_, vec X_) { //removed vec theta
  
  arma::vec y = y_;
  mat X(y.n_elem, 1, fill::ones);
  X.insert_cols(1, X_);
 
  // double alpha0 = *theta0;
  // double alpha1 = *theta1;
  // double alpha2 = *theta2;

  double alpha0 = theta[0];
  double alpha1 = theta[1];
  double alpha2 = theta[2];
  // double* alpha0 = theta[0];
  // double* alpha1 = theta[1];
  // double* alpha2 = theta[2];


  int num_observations = y.n_elem;
  int num_variables = X.n_cols;

  arma::mat F;
  F.eye(num_variables, num_variables);

  double R = pow(alpha0, 2);

  // Q is a kxk matrix of parameters we must estimate:  I have made it diagonal.
  // use the below when we improve
  // arma::mat Q = arma::mat(num_variables, num_variables, fill::zeros);
  // with the translated R diag(Q) <- c(theta[2]^2, theta[3]^2)
  
  // TODO: need to adjust this manually at this point
  
  arma::mat Q;
  Q << alpha1 << 0 << arma::endr
    << 0 << alpha2 << arma::endr;

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

	  //was this but i think it must be wrong, we want 2x2 matrix
	  // arma::vec as_vector_tempPtt_1(4);
	  // as_vector_tempPtt_1.at(0) =  Ptt_1.row(i)[0];
	  // as_vector_tempPtt_1.at(1) =  Ptt_1.row(i)[1];
	  // as_vector_tempPtt_1.at(2) =  Ptt_1.row(i)[2];
	  // as_vector_tempPtt_1.at(3) =  Ptt_1.row(i)[3];

	  arma::mat as_mat_tempPtt_1(2, 2);
	  as_mat_tempPtt_1 << Ptt_1.row(i)[0] << Ptt_1.row(i)[2] << arma::endr
			   << Ptt_1.row(i)[1] << Ptt_1.row(i)[3] << arma::endr;

	  betatt.row(i) = betatt_1.row(i) + (((as_mat_tempPtt_1 * X.row(i).t()) * eta.at(i)) * (1 / f.at(i))).t();

	  Ptt.row(i) = arma::vectorise(as_mat_tempPtt_1 - as_mat_tempPtt_1 * X.row(i).t() * X.row(i) * as_mat_tempPtt_1 * (1 / f.at(i))).t();

  }

  //the first element of f is 0, so change to 1---log(1) is 0 and 1/1 is 1, so OK
  f.at(0) = 1;

  double logl = -0.5 * arma::accu(log(abs(f))) - 0.5 * arma::accu(eta % eta * (1 / f).t());

  return -1 * logl;

    }


int main(int argc, char** argv)
  {
    //import market and stock--------
    colvec stock;
    stock.load("./stock.csv", csv_ascii);

    colvec market;
    market.load("./market.csv", csv_ascii);

    //    vec theta(3, fill::randu);
    // double theta0 = theta[0];
    // double theta1 = theta[1];
    // double theta2 = theta[2];

    double theta[3] = {0.1, 0.1, 0.1};

    // colvec y_ = stock;
    // colvec X_ = market;
    // colvec theta(3, fill::randu);

 //double lik_test = lik(theta, stock, market);
 
    //     cout << "neg log lik = " << lik(&theta0, &theta1, &theta2, stock, market) << endl;
    //    cout << "neg log lik = " << lik(&theta, stock, market) << endl;

    //can use function minimization from the GSL here: https://www.gnu.org/software/gsl/doc/html/multimin.html#
    //to implement it this answer may be useful: https://stackoverflow.com/questions/62264648/using-gsl-minimize-in-c
    //BETTER: Use NLOpt

     //minimization

     nlopt_opt opt;
     opt = nlopt_create(NLOPT_LD_MMA, 3); /* algorithm and dimensionality */

     double minf; /* `*`the` `minimum` `objective` `value,` `upon` `return`*` */
     if (nlopt_optimize(opt, theta, &minf) < 0) {
    printf("nlopt failed!\n");
}
else {
  printf("found minimum at f(%g,%g,%g) = %0.10g\n", theta[0], theta[1], theta[2], minf);
  }

  }
