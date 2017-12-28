/*
  Randomized SVD implementation with Eigen

  Based on:
  * fast.ai Numerical Linear Algebra course

*/

#include <chrono>
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "utils.h"

using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixBase;

// TODO
// * understand:
//     * build intuition for U and V matrices
//     * write down how to do inverse
//     * difference between pca and svd
//     * read papers, figure out what power iterations are for
// * check for edge cases (matrix dimensions vs rank, oversamples)

// * speed up - parallelize with Eigen's parallel LU, manually parallelize
// QR?
// * Use Eigen::Ref to make more general? But this need to be float or double anyways, and Matrixf or Matrixd can be cast to MatrixXd? What about static matrices?

// interface is same as Eigen SVD
class RandomizedSvd {
 public:
  RandomizedSvd(const MatrixXd& m, int rank, int oversamples = 10, int iter = 2)
      : U_(), V_(), S_() {
    ComputeRandomizedSvd(m, rank, oversamples, iter);
  }

  VectorXd singularValues() { return S_; }
  MatrixXd matrixU() { return U_; }
  MatrixXd matrixV() { return V_; }

 private:
  MatrixXd U_, V_;
  VectorXd S_;

  void ComputeRandomizedSvd(const MatrixXd& A, int rank, int oversamples,
                            int iter) {

    using namespace std::chrono;
    auto start = steady_clock::now();

    // TODO account for skinny and fat A matrix: check dimensions

    // Add some additional samples for accuracy
    MatrixXd Q = FindRandomizedRange(A, rank + oversamples, iter);
    auto now = steady_clock::now();
    long int elapsed = duration_cast<milliseconds>(now - start).count();
    MatrixXd B = Q.transpose() * A;

    // Compute the SVD on the thin matrix (much cheaper than SVD on original)
    start = steady_clock::now();
    Eigen::JacobiSVD<MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    now = steady_clock::now();
    elapsed = duration_cast<milliseconds>(now - start).count();

    U_ = (Q * svd.matrixU()).block(0, 0, A.rows(), rank);
    V_ = svd.matrixV().transpose().block(0, 0, rank, A.cols());
    S_ = svd.singularValues().head(rank);
  }

  /*
    Finds a set of orthonormal vectors that approximates the range of A
    Basic idea is that finding orthonormal basis vectors for AW, where w is some
    random vectors, can approximate the range of A
    Most of the time/computation in the randomized SVD is spent here
  */
  MatrixXd FindRandomizedRange(const MatrixXd& A, int size, int iter) {
    int nr = A.rows(), nc = A.cols();
    MatrixXd L(nr, size);
    Eigen::FullPivLU<MatrixXd> lu1(nr, size);
    MatrixXd Q = MatrixXd::Random(nc, size); // TODO should this be stack or dynamic allocation?
    Eigen::FullPivLU<MatrixXd> lu2(nc, nr);

    // Conduct normalized power iterations
    // From Facebook implementation: "Please note that even n_iter=1 guarantees superb accuracy, whether or not there is any gap in the singular values of the matrix A being approximated"
    for (int i = 0; i < iter; ++i) {
      lu1.compute(A * Q);
      L.setIdentity();
      L.block(0, 0, nr, size).triangularView<Eigen::StrictlyLower>() =
          lu1.matrixLU();

      lu2.compute(A.transpose() * L);
      Q.setIdentity();
      Q.block(0, 0, nc, size).triangularView<Eigen::StrictlyLower>() =
          lu2.matrixLU();
    }

    Eigen::ColPivHouseholderQR<MatrixXd> qr(A * Q);
    return qr.householderQ() * MatrixXd::Identity(nr, size); // recover skinny Q matrix
  }
};

/*
  Computes spectral norm of error in reconstruction, from SVD matrices.

  Spectral norm = square root of maximum eigenvalue of matrix. Intuitively: the maximum 'scale', by which a matrix can 'stretch' a vector.
  Note: The definition of an eigenvalue is for square matrices. For non square matrices, we can define singular values: Definition: The singular values of a m×n matrix A are the positive square roots of the nonzero eigenvalues of the corresponding matrix A'A. The corresponding eigenvectors are called the singular vectors.
*/
double diff_spectral_norm(MatrixXd& A, MatrixXd& U, VectorXd& s, MatrixXd& V, int n_iter=20) {
  int nr = A.rows(), nc = A.cols(), snorm;

  VectorXd y = VectorXd::Random(nr);
  y.normalize();
  MatrixXd S = s.asDiagonal(); // TODO use diagonal-ness to speed up computation?
  // TODO pre-calculate S.inverse()?
  VectorXd x(nr);

  // TODO implement and compare fbpca's method
  // TODO figure out ways to make this more efficient (memory, compute)
  // Run n iterations of the power method
  MatrixXd B = (A - U*S*V);

  if(B.rows() != B.cols())
     B = B*B.transpose();

  for(int i=0; i<n_iter; ++i) {
    y = B*y;
    y.normalize();
  }
  double eigval = abs((B*y).dot(y) / y.dot(y));

  return sqrt(eigval);
}

// Test for finding diff_spectral_norm
double find_largest_eigenvalue(MatrixXd& A) {
  int n_iter = 20;

  VectorXd y = VectorXd::Random(A.rows());

  // Method for non-square matrices
  // double snorm;
  // TODO understand and fix this
  // for(int i=0; i<n_iter; ++i){
  //   VectorXd x = A.transpose()*y;
  //   y = A*x;
  //   snorm = y.norm();
  //   y.normalize();
  // }
  //
  // cout << "Normalized eigenvector: " << y.transpose() << endl;
  // double eigval = sqrt(snorm);

  // Simple method for square matrices - extended for nonsquare
  double eigval;
  if(A.rows() == A.cols()) {
    for(int i=0; i<n_iter; ++i) {
      y = A*y;
      y.normalize();
    }
    eigval = (A*y).dot(y) / y.dot(y);
  }
  else {
    for(int i=0; i<n_iter; ++i) {
      y = A*A.transpose()*y;
      y.normalize();
    }
    eigval = (A*A.transpose()*y).dot(y) / y.dot(y);
  }
  // eigval = (eigval >= 0) ? eigval : -eigval;

  cout << "Normalized eigenvector: " << y.transpose() << endl;

  return eigval;
}
