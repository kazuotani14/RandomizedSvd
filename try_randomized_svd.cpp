#include "randomized_svd.h"

using namespace std;
using namespace std::chrono;

int main() {
  Eigen::MatrixXd M = Eigen::MatrixXd(5000, 1000);
  M.setRandom();


  // Full SVD
  cout << "Full SVD: " << endl;
  auto start = steady_clock::now();
  Eigen::JacobiSVD<MatrixXd> full_svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);
  auto now = steady_clock::now();
  long int elapsed = duration_cast<milliseconds>(now - start).count();
  cout << elapsed << " ms" << endl;

  // Do randomized SVD
  cout << "Randomized SVD: " << endl;
  int rank = 5;
  start = steady_clock::now();
  RandomizedSvd rsvd(M, rank);
  now = steady_clock::now();
  elapsed = duration_cast<milliseconds>(now - start).count();
  std::cout << elapsed << " ms" << std::endl;

  MatrixXd U = full_svd.matrixU();
  MatrixXd V = full_svd.matrixV();
  VectorXd s = full_svd.singularValues();
  MatrixXd S = s.asDiagonal();

  // MatrixXd U = rsvd.matrixU();
  // MatrixXd V = rsvd.matrixV();
  // VectorXd s = rsvd.singularValues();
  // MatrixXd S = s.asDiagonal();
  cout << "Reconstruction error (spectral norm): " << diff_spectral_norm(M, U, s, V) << endl;;

  /* Testing power method */

  // Eigen::MatrixXd M(2, 2); M << 2, -12, 1, -5;
  // Eigen::MatrixXd M(2, 3); M << 2, -12, 1, -5, 3, 5;
  // cout << M << endl;
  // cout << find_largest_eigenvalue(M) << endl;

  // using VectorUd = Eigen::Matrix<unsigned char, Eigen::Dynamic, 1>;
  // VectorUd v(3);
  // v << 0, 254, 15;
  // VectorXd v2 = v.cast<double>();
  // cout << v.transpose() << endl;
  // cout << v2.transpose() << endl;
  //
  // VectorUd v3 = v2.cast<unsigned char>();
  // cout << (v==v3) << endl;

  return 0;
}
