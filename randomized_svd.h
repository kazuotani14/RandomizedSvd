/*
  Randomized SVD implementation with Eigen, based on fast.ai Numerical Linear Algebra course
*/

#include "Eigen/Dense"
#include <chrono>
#include <iostream>

using std::cout;
using std::endl;
using Eigen::MatrixXd;
using Eigen::VectorXd;

Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "", "");

void print_eigen(const std::string name, const Eigen::MatrixXd& mat) {
  if (mat.cols() == 1) {
    std::cout << name << ": " << mat.transpose().format(CleanFmt) << std::endl;
  } else {
    std::cout << name << ":\n" << mat.format(CleanFmt) << std::endl;
  }
}

// TODO speed up - parallelize with Eigen's parallel LU, parallelize QR?
class RandomizedSvd {

public:
  RandomizedSvd(const MatrixXd& m, int rank, int oversamples=2, int iter=3)
    : U_(), V_(), S_() {
    ComputeRandomizedSvd(m, rank, oversamples, iter);
  }

  VectorXd singularValues() { return S_; }
  MatrixXd matrixU() { return U_; }
  MatrixXd matrixV() { return V_; }

  void compute(const MatrixXd& m);

private:
  MatrixXd U_, V_;
  VectorXd S_;

  void ComputeRandomizedSvd(const MatrixXd& A, int rank, int oversamples, int iter) {
    // Find orthonormal vectors that approximates range of A
    MatrixXd Q = FindRandomizedRange(A, rank+oversamples, iter);
    MatrixXd B = Q.transpose()*A;

    // Compute the SVD on the thin matrix
    Eigen::JacobiSVD<MatrixXd> svd(B, Eigen::ComputeThinU | Eigen::ComputeThinV);
    U_ = (Q * svd.matrixU()).block(0, 0, A.rows(), rank);
    V_ = svd.matrixV().transpose().block(0, 0, rank, A.cols());
    S_ = svd.singularValues().head(rank);
  }

  MatrixXd FindRandomizedRange(const MatrixXd& A, int size, int iter) {
    int nr = A.rows(), nc = A.cols();
    MatrixXd Q = MatrixXd::Random(nc, size);
    MatrixXd L(nr, size);
    Eigen::FullPivLU<MatrixXd> lu1(nr, size);
    Eigen::FullPivLU<MatrixXd> lu2(nc, nr);

    for(int i=0; i<iter; ++i) {
      lu1.compute(A*Q);
      L.setIdentity();
      L.block(0,0, nr, size).triangularView<Eigen::StrictlyLower>() = lu1.matrixLU();

      lu2.compute(A.transpose()*L);
      Q.setIdentity();
      Q.block(0,0, nc, size).triangularView<Eigen::StrictlyLower>() = lu2.matrixLU();
    }

    Eigen::ColPivHouseholderQR<MatrixXd> qr(A*Q);
    return qr.householderQ();
  }

};

/*
# computes an orthonormal matrix whose range approximates the range of A
# power_iteration_normalizer can be safe_sparse_dot (fast but unstable), LU (imbetween), or QR (slow but most accurate)
def randomized_range_finder(A, size, n_iter=5):
    Q = np.random.normal(size=(A.shape[1], size))
#     print('init:', Q.shape)

    for i in range(n_iter):
        Q, _ = linalg.lu(A @ Q, permute_l=True)
        #print('1:', Q.shape)
        Q, _ = linalg.lu(A.T @ Q, permute_l=True) # Q is U! Figure out what linalg.lu returns
        #print('2:', Q.shape)
        #print(orthonormal_check(Q))
        # Note: for LU of non-square matrices, matrix on side of smaller dim will be square

    Q, _ = linalg.qr(A @ Q, mode='economic')
    return Q

def randomized_svd(M, n_components, n_oversamples=10, n_iter=3):

    # Oversamples are for a bit more accuracy
    n_random = n_components + n_oversamples

    Q = randomized_range_finder(M, n_random, n_iter)

    # project M to the (k + p) dimensional space using the basis vectors
    B = Q.T @ M

    # compute the SVD on the thin matrix: (k + p) wide
    Uhat, s, V = linalg.svd(B, full_matrices=False)
    del B #What is this for?
    U = Q @ Uhat

    return U[:, :n_components], s[:n_components], V[:n_components, :]

*/
