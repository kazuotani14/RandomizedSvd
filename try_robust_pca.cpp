#include "robust_pca.h"
#include <chrono>
#include <thread>

int main(int /*argc*/, char** /*argv*/) {

  MatrixXd test(2,2);
  test << 2, 0, 0.1, 0.5;

  // cout << test << endl;

  // Subtracting scalar from a matrix
  // test = test - 1*MatrixXd::Ones(test.rows(), test.cols());

  // Eigen equivalent of np.where
  test = (test.array() >= 0.5).select(
    MatrixXd::Ones(test.rows(), test.cols()),
    MatrixXd::Zero(test.rows(), test.cols()));

  // cout << test << endl;;
  // cout << (test.array() > 0.5).count() << endl;

  MatrixXd M(6,3);
  // M << 1, 0, 0.02,
  //      1, 0.1, 0.05,
  //      1, -0.1, 0.03,
  //      1, 0.3, -0.01,
  //      1, -0.2, 0.01,
  //      1, 0, 0.02;

 M << 1, 1, 1,
      0.01, 0.1, 0.05,
      0.2, -0.1, 0.03,
      0.01, 0.3, -0.01,
      0.03, -0.2, 0.01,
      0, 0, 0.02;

  RobustPca rpca(M);

  cout << "M:\n" << M << endl;
  cout << "L:\n" << rpca.LowRankComponent() << endl;
  cout << "S:\n" << rpca.SparseComponent() << endl;

  // cout << "\n\ntest" << std::flush;
  // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // cout << "\r" << std::flush;
  // std::this_thread::sleep_for(std::chrono::milliseconds(1000));
  // cout << "test2";

  return 0;
}
