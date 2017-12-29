/*
  Testing background removal from webcam feed using randomized SVD.

  Idea is to run rSVD on past N frames to extract background, then subtract that out of image.
  Assumes static background, moving foreground.
  Doesn't work that well at this point.

  TODO
   * clean up code, add comments/documentation
   * look at/test other methods for background subtraction
     * mean/median/mode, thresholding
     * low-pass filter (smooth out small variations in lighting)
     * robust PCA
     * https://docs.opencv.org/3.1.0/d1/dc5/tutorial_background_subtraction.html
*/

#include <chrono>
#include <iostream>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include "Eigen/Dense"
#include "randomized_svd.h"

using std::cout;
using std::endl;
using uchar = unsigned char;
using MatrixUd = Eigen::Matrix<uchar, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>;
using VectorUd = Eigen::Matrix<uchar, Eigen::Dynamic, 1>;

// Number of frames to hold in frame buffer (for computing low-rank approximation)
const int sliding_window = 100;

// Image is resized to be smaller, for efficiency
const int n_rows = 120;
const int n_cols = 213;
cv::Size size(n_cols, n_rows);

/*
  A bunch of conversions in one function...
  * Resize image to be smaller
  * Convert RGB to grayscale
  * Convert OpenCV Mat to Eigen Matrix
*/
void OpencvRgb2EigenGreyscaleSmall(cv::Mat& A, MatrixUd& B) {
  cv::resize(A, A, size);
  cv::cvtColor(A, A, CV_BGR2GRAY);
  // TODO get rid of copy
  Eigen::Map<MatrixUd> C(A.ptr<uchar>(), A.rows, A.cols);
  B = C;
}


int main() {
  // Open camera feed
  cv::VideoCapture cam(0);
  if (!cam.isOpened()) return -1;
  cv::namedWindow("camera", 1);

  cv::Mat frame;
  MatrixUd eigen_frame;

  // Fill ring buffer of frames (for computing low-rank approximation/background)
  cout << "Filling ring buffer." << endl;
  MatrixUd frame_buf(sliding_window, n_rows * n_cols);
  int buf_idx = 0;
  for (int i = 0; i < sliding_window; ++i) {
    cam.read(frame);
    OpencvRgb2EigenGreyscaleSmall(frame, eigen_frame);

    // Flatten and fill buffer row
    VectorUd flat_frame = Eigen::Map<VectorUd>(eigen_frame.data(), eigen_frame.size());
    frame_buf.row((buf_idx+1) % sliding_window) = flat_frame;
  }

  cout << "Webcam window open." << endl;
  while (true) {
    cam.read(frame);

    OpencvRgb2EigenGreyscaleSmall(frame, eigen_frame);

    // Flatten current frame and add to rolling buffer
    VectorUd flat_frame = Eigen::Map<VectorUd>(eigen_frame.data(), eigen_frame.size());
    frame_buf.row(buf_idx) = flat_frame;

    // Take randomized svd to find low-rank approximation (which should be background)
    // TODO preallocate rsvd and recompute?
    // TODO just get first singular vector and subtract that from current frame
    RandomizedSvd rsvd(frame_buf.cast<double>(), 1);
    MatrixXd U = rsvd.matrixU();
    MatrixXd V = rsvd.matrixV();
    VectorXd s = rsvd.singularValues();
    MatrixXd S = s.asDiagonal();
    MatrixUd low_rank = (U * S * V.transpose()).cast<unsigned char>();

    // Remove background from all images in frame buffer, take current frame
    VectorUd bg_removed = (frame_buf - low_rank).row(buf_idx);
    VectorUd bg = low_rank.row(buf_idx);

    // Reconstruct opencv mat with "foreground" image
    cv::Mat bg_removed_mat(eigen_frame.rows(), eigen_frame.cols(), CV_8U, bg_removed.data());
    cv::Mat bg_mat(eigen_frame.rows(), eigen_frame.cols(), CV_8U, bg.data());

    // cout << (int)bg_removed.minCoeff() << ", " << (int)bg_removed.maxCoeff() << endl;
    // double min, max;
    // cv::minMaxLoc(new_frame, &min, &max);
    // cout << min << ", " << max << endl;

    // show current frame, background, and "foreground" image side-by-side
    cv::Mat sidebyside(cv::Size(frame.cols*3, frame.rows), frame.type(), cv::Scalar::all(0));
    frame.copyTo(sidebyside(cv::Rect(0, 0, n_cols, n_rows)));
    bg_mat.copyTo(sidebyside(cv::Rect(n_cols, 0, n_cols, n_rows)));
    bg_removed_mat.copyTo(sidebyside(cv::Rect(n_cols*2, 0, n_cols, n_rows)));
    cv::imshow("frame, background, foreground", sidebyside);

    buf_idx = (buf_idx+1) % sliding_window;
    if (cv::waitKey(1) >= 0) break;
  }

  return 0;
}
