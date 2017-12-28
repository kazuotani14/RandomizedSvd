#include "randomized_svd.h"

using namespace std;
using namespace std::chrono;

/*
  Test randomized SVD by computing decomposition of large (random) matrix

  Empirically, I found that increasing the rank of the randomized SVD doesn't decrease the reconstruction error enough to make it worth the increase computation. You can check that the algorithm works by setting rank_rsvd=rank(M)
*/
int main() {
  Eigen::MatrixXd M = Eigen::MatrixXd(1000, 500);
  srand((unsigned int) time(0));
  M.setRandom();

  // Full SVD
  cout << "Full SVD: ";
  auto start = steady_clock::now();

  Eigen::JacobiSVD<MatrixXd> full_svd(M, Eigen::ComputeThinU | Eigen::ComputeThinV);

  auto now = steady_clock::now();
  long int elapsed = duration_cast<milliseconds>(now - start).count();
  cout << elapsed << " ms" << endl;

  // Randomized SVD
  int rank = 10;
  cout << "Randomized SVD with rank " << rank << ": ";
  start = steady_clock::now();

  RandomizedSvd rsvd(M, rank);

  now = steady_clock::now();
  elapsed = duration_cast<milliseconds>(now - start).count();
  std::cout << elapsed << " ms" << std::endl;

  cout << "Reconstruction error for full SVD (zero): " <<
    diff_spectral_norm(M, full_svd.matrixU(), full_svd.singularValues(), full_svd.matrixV()) << endl;
  cout << "Reconstruction error for rand SVD: " <<
    diff_spectral_norm(M, rsvd.matrixU(), rsvd.singularValues(), rsvd.matrixV()) << endl;

  return 0;
}
