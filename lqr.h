#include <Eigen/Dense>

template <size_t m, size_t n>
class LQR {
 private:
  Eigen::Matrix<double, m, m> A;
  Eigen::Matrix<double, m, n> B;

  Eigen::Matrix<double, m, m> solveSchur(const Eigen::Matrix<double, 2 * m, 2 * m> Z) const;

 public:
  void setState(const Eigen::Matrix<double, m, m>& A, const Eigen::Matrix<double, m, n>& B);
  Eigen::Matrix<double, m, n> computeK(const Eigen::Matrix<double, m, m>& Q,
                                       const Eigen::Matrix<double, n, n>& R) const;
  bool isControllable() const;
};