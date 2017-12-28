#ifndef _UTILS_H_
#define _UTILS_H_

#include "Eigen/Dense"

Eigen::IOFormat CleanFmt(3, 0, ", ", "\n", "", "");

void print_eigen(const std::string name, const Eigen::MatrixXd& mat) {
  if (mat.cols() == 1) {
    std::cout << name << ": " << mat.transpose().format(CleanFmt) << std::endl;
  } else {
    std::cout << name << ":\n" << mat.format(CleanFmt) << std::endl;
  }
}

#endif
