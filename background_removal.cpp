#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Eigen/Dense"
#include "randomized_svd.h"

#include <iostream>
using std::cout;
using std::endl;

using uchar = unsigned char;
using MatrixUd = Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorUd = Eigen::Matrix<uchar, Eigen::Dynamic, 1>;

const int sliding_window = 100;
const int n_rows = 120;  // TODO change these to maintain aspect ratio
const int n_cols = 213;
cv::Size size(n_cols, n_rows);

// TODO
// * clean up code, add comments/documentation
// * look at/test other methods for background subtraction

// TODO
void OpencvRgb2EigenGreyscaleSmall(cv::Mat& A, MatrixUd& B) {
  // Resize image to reasonable size. TODO check if we actually need this step
  cv::resize(A, A, size);

  cv::cvtColor(A, A, CV_BGR2GRAY);

  // TODO get rid of copy
  Eigen::Map<MatrixUd>
    C(A.ptr<uchar>(), A.rows, A.cols);
  B = C;
}


// SVD method assumes that camera/background is static


int main() {
  cv::VideoCapture cam(0);
  if (!cam.isOpened()) return -1;

  // TODO figure out why this isn't working
  cv::namedWindow("camera", 1);

  cv::Mat frame;
  MatrixUd eigen_frame;

  int buf_idx = 0;
  MatrixUd frame_buf(sliding_window, n_rows * n_cols);

  // Fill ring buffer of frames
  cout << "Filling ring buffer." << endl;
  for (int i = 0; i < sliding_window; ++i) {
    cam.read(frame);

    // resize, convert to greyscale, convert to eigen matrix
    OpencvRgb2EigenGreyscaleSmall(frame, eigen_frame);

    // Flatten and fill buffer row
    VectorUd flat_frame = Eigen::Map<VectorUd>(eigen_frame.data(), eigen_frame.size());
    frame_buf.row((buf_idx+1) % sliding_window) = flat_frame;
  }

  // Eigen::MatrixXd eigen_frame(213, 120);

  cout << "Webcam window open." << endl;
  while (true) {
    cam.read(frame);

    // resize, convert to greyscale, convert to eigen matrix
    OpencvRgb2EigenGreyscaleSmall(frame, eigen_frame);

    // TODO show original image, low-rank approx, and subtracted in one window

    // Add current frame to rolling buffer
    VectorUd flat_frame = Eigen::Map<VectorUd>(eigen_frame.data(), eigen_frame.size());
    frame_buf.row(buf_idx) = flat_frame;

    // Take randomized svd to find low-rank approximation (hopefully background)
    // TODO preallocate rsvd and recompute?
    // TODO just get first singular vector and subtract that from current frame
    RandomizedSvd rsvd(frame_buf.cast<double>(), 1);
    MatrixXd U = rsvd.matrixU();
    MatrixXd V = rsvd.matrixV();
    VectorXd s = rsvd.singularValues();
    MatrixXd S = s.asDiagonal();
    MatrixUd low_rank = (U * S * V).cast<unsigned char>();

    // Remove background from current frame
    // TODO try to smooth out lighting conditions
    VectorUd bg_removed = (frame_buf - low_rank).row(buf_idx);
    // cout << (int)bg_removed.minCoeff() << ", " << (int)bg_removed.maxCoeff() << endl;

    // cout << low_rank.rows() << " " << low_rank.cols() << ", " << frame_buf.rows() << " " <<  frame_buf.cols() << endl;

    // Reconstruct opencv mat and show
    cv::Mat new_frame(eigen_frame.rows(), eigen_frame.cols(), CV_8U, bg_removed.data());
    // double min, max;
    // cv::minMaxLoc(new_frame, &min, &max);
    // cout << min << ", " << max << endl;


    // show current frame and "foreground" image side-by-side
    cv::Mat sidebyside(cv::Size(frame.cols*2, frame.rows), frame.type(), cv::Scalar::all(0));
    frame.copyTo(sidebyside(cv::Rect(0, 0, n_cols, n_rows)));
    new_frame.copyTo(sidebyside(cv::Rect(n_cols, 0, n_cols, n_rows)));
    cv::imshow("cam", sidebyside);

    buf_idx = (buf_idx+1) % sliding_window;
    if (cv::waitKey(1) >= 0) break;
  }

  return 0;
}
