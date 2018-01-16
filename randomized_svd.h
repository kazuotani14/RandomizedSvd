/*
  Randomized SVD for fast approximate matrix decomposition
  Interface is same as Eigen's jacobiSVD

  TODO
  * account for skinny and fat matrices
  * speed up: parallelize with Eigen's parallel LU, manually parallelize QR?
  * Use Eigen::Ref to make more general? But this need to be float or double anyways, and Matrixf or Matrixd can be cast to MatrixXd? What about static matrices?
*/

#ifndef _RANDOMIZEDSVD_H_
#define _RANDOMIZEDSVD_H_

#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "utils.h"

using std::cout;
using std::endl;
using std::min;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixBase;


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

  /*
    Main function for randomized svd
    oversamples: additional samples/rank for accuracy, to account for random sampling
  */
  void ComputeRandomizedSvd(const MatrixXd& A, int rank, int oversamples,
                            int iter) {
    using namespace std::chrono;

    // If matrix is too small for desired rank/oversamples
    if((rank + oversamples) > min(A.rows(), A.cols())) {
      rank = min(A.rows(), A.cols());
      oversamples = 0;
    }

    MatrixXd Q = FindRandomizedRange(A, rank + oversamples, iter);
    MatrixXd B = Q.transpose() * A;

    // Compute the SVD on the thin matrix (much cheaper than SVD on original)
    Eigen::JacobiSVD<MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);

    U_ = (Q * svd.matrixU()).block(0, 0, A.rows(), rank);
    V_ = svd.matrixV().block(0, 0, A.cols(), rank);
    S_ = svd.singularValues().head(rank);
  }

  /*
    Finds a set of orthonormal vectors that approximates the range of A
    Basic idea is that finding orthonormal basis vectors for A*W, where W is set of some
    random vectors w_i, can approximate the range of A
    Most of the time/computation in the randomized SVD is spent here
  */
  MatrixXd FindRandomizedRange(const MatrixXd& A, int size, int iter) {
    int nr = A.rows(), nc = A.cols();
    MatrixXd L(nr, size);
    Eigen::FullPivLU<MatrixXd> lu1(nr, size);
    MatrixXd Q = MatrixXd::Random(nc, size); // TODO should this be stack or dynamic allocation?
    Eigen::FullPivLU<MatrixXd> lu2(nc, nr);

    // Conduct normalized power iterations
    // Intuition: multiply by A a few times to find a matrix Q that's "more in the range of A"
    //  Simply multiplying by A repeatedly makes alg unstable, so use LU to "normalize"
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
double diff_spectral_norm(MatrixXd A, MatrixXd U, VectorXd s, MatrixXd V, int n_iter=20) {
  int nr = A.rows();

  VectorXd y = VectorXd::Random(nr);
  y.normalize();

  MatrixXd B = (A - U*s.asDiagonal()*V.transpose());

  // TODO make this more efficient (don't explicitly calculate B)
  if(B.rows() != B.cols())
     B = B*B.transpose();

 // Run n iterations of the power method
 // TODO implement and compare fbpca's method
  for(int i=0; i<n_iter; ++i) {
    y = B*y;
    y.normalize();
  }
  double eigval = abs((B*y).dot(y) / y.dot(y));
  if(eigval==0) return 0;

  return sqrt(eigval);
}

#endif
