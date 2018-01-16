/*
  Implementation of Robust PCA algorithm via Principal Component Pursuit
  Separates a matrix into two-components: low-rank and sparse
  The sparse component can separate "noise"

  Based on: Candes, E. J., Li, X., Ma, Y., & Wright, J. (2009). Robust Principal Component Analysis.

TODO
* Make sure this works - test against fast.ai implementation
* Performance optimizations
*/

#ifndef _ROBUSTPCA_H_
#define _ROBUSTPCA_H_

#include <algorithm>
#include <chrono>
#include <iostream>
#include <cmath>
#include "Eigen/Dense"
#include "randomized_svd.h"

using std::cout;
using std::endl;
using std::max;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using Eigen::MatrixBase;
using namespace std::chrono;

const double TOL = 1e-9;

class RobustPca {
public:
  RobustPca(MatrixXd M, int maxiter=10, int k=10) : L(), S() {
    ComputeRobustPca(M, maxiter, k);
  }

  MatrixXd LowRankComponent() { return L; }
  MatrixXd SparseComponent() { return S; }

private:
  MatrixXd L, S;
  double nr, nc;

  void ComputeRobustPca(MatrixXd& M, int maxiter, int k) {
    nr = M.rows();
    nc = M.cols();
    bool trans = (nr < nc);

    if (trans) {
      M.transposeInPlace();
      nr = M.rows();
      nc = M.cols();
    }

    double lambda = 1/sqrt(nr);
    double op_norm = norm_op(M);

    double init_scale = max(op_norm, M.lpNorm<Eigen::Infinity>()) * lambda;
    MatrixXd Y = M /init_scale;
    MatrixXd Z;

    double mu = k*1.25/op_norm;
    double mu_bar = mu*1e7;
    double rho = k * 1.5;

    double d_norm = M.norm();
    L = MatrixXd::Zero(nr, nc);
    double sv = 1;

    for(int i=0; i<maxiter; ++i){

      cout << "rank sv: " << sv << endl;
      MatrixXd M2 = M + Y/mu;

      auto start = steady_clock::now();

      /* update estimate of sparse component */
      shrink(M2 - L, lambda/mu, S);

      auto now = steady_clock::now();
      long int elapsed = duration_cast<milliseconds>(now - start).count();
      // std::cout << "shrink: " << elapsed << std::endl;

      start = steady_clock::now();

      /* update estimate of low-rank component */
      int svp = svd_truncate(M2 - S, sv, 1/mu, L);
      sv = svp + ((svp < sv) ? 1 : round(0.05*nc));

      now = steady_clock::now();
      elapsed = duration_cast<milliseconds>(now - start).count();
      // std::cout << "svd_truncate: " << elapsed << std::endl;

      start = steady_clock::now();

      /* compute residual */
      Z = M - L - S;
      Y = mu * Z;
      mu *= rho;

      if (mu > mu_bar)
        mu = mu_bar;
      if (converged(Z, d_norm)) {
        cout << "converged!" << endl;
        break;
      }

      now = steady_clock::now();
      elapsed = duration_cast<milliseconds>(now - start).count();
      // std::cout << "residual, etc: " << elapsed << std::endl;
    }

    if (trans) {
      L.transposeInPlace();
      S.transposeInPlace();
    }
  }

  // Encourages sparsity in M by slightly shrinking all values, thresholding small values to zero
  void shrink(const MatrixXd& M, double tau, MatrixXd& S) {
    S = M - tau*MatrixXd::Ones(nr, nc);
    S = (M.array() > 0).select(M, MatrixXd::Zero(nr, nc));
  }

  // Encourages low-rank by taking (truncated) SVD, then setting small singular values to zero
  int svd_truncate(const MatrixXd& M, int rank, double min_sv, MatrixXd& L) {
    RandomizedSvd rsvd(M, rank);

    VectorXd s = rsvd.singularValues();
    s = s - min_sv*VectorXd::Ones(s.size());
    int nnz = (s.array() > 0).count();

    L = rsvd.matrixU().leftCols(nnz) * s.head(nnz).asDiagonal() * rsvd.matrixV().transpose().topRows(nnz);

    return nnz;
  }

  bool converged(const MatrixXd& Z, double d_norm) {
    double err = Z.norm() / d_norm;
    cout << "error: " << err << endl;
    return (err < TOL);
  }

  // Returns first/largest singular value of M
  double norm_op(const MatrixXd& M, int rank=5) {
    RandomizedSvd rsvd(M, rank);
    VectorXd s = rsvd.singularValues();
    return rsvd.singularValues()[0];
  }

};

#endif
