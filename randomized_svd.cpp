#include "randomized_svd.h"
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

int main()
{
    // cv::VideoCapture cam(0);
    // if(!cam.isOpened())
    //   return -1;
    //
    // cv::Mat frame;
    // while(true) {
    //   cam.read(frame);
    //   cv::imshow("cam", frame);
    //   if(cv::waitKey(30) >= 0) break;
    // }

    int rank = 2;

    using Mat = Eigen::Matrix<double, 10, 5>;
    Mat M = Mat::Random();

    // MatrixXd m(3,3);
    // m << 1, 2, 3, 4, 5, 6, 7, 8, 9;

    RandomizedSvd rsvd(M, rank);
    MatrixXd U = rsvd.matrixU();
    MatrixXd V = rsvd.matrixV();
    VectorXd s = rsvd.singularValues();

    // TODO figure out equivalent to diag(vec)
    MatrixXd S(s.size(), s.size());
    S.setZero();
    for(int i=0; i<s.size(); ++i) {
      S(i,i) = s(i);
    }

    MatrixXd reconstruction = U*S*V;

    print_eigen("original", M);
    print_eigen("reconstruction", reconstruction);

    return 0;
}
