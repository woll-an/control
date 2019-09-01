#include <Eigen/Dense>

template <size_t m, size_t n>
class LQR {
 private:
  Eigen::Matrix<double, m, m>& A;
  Eigen::Matrix<double, m, n>& B;

 public:
  LQR(Eigen::Matrix<double, m, m>& A, Eigen::Matrix<double, m, n>& B);
  Eigen::Matrix<double, m, n> computeK(Eigen::Matrix<double, m, m>& Q,
                                       Eigen::Matrix<double, n, n>& R);
  bool isControllable();
};