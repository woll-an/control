#include <stdexcept>

#include "lqr.h"

template <size_t m, size_t n>
void LQR<m, n>::setState(const Eigen::Matrix<double, m, m>& A_in,
                         const Eigen::Matrix<double, m, n>& B_in) {
  A = A_in;
  B = B_in;
}

template <size_t m, size_t n>
Eigen::Matrix<double, m, n> LQR<m, n>::computeK(const Eigen::Matrix<double, m, m>& Q,
                                                const Eigen::Matrix<double, n, n>& R) const {
  if (!isControllable()) {
    throw std::invalid_argument("system is not controllable");
  }

  if (R.determinant() == 0) {
    throw std::invalid_argument("R cannot be inverted");
  }

  Eigen::Matrix<double, 2 * m, 2 * m> Z;
  Z << A, -B * R.inverse() * B.transpose(), -Q, -A.transpose();
  Eigen::Matrix<double, m, m> P = solveSchur(Z);

  return R.inverse() * (B.transpose() * P);
};

template <size_t m, size_t n>
bool LQR<m, n>::isControllable() const {
  Eigen::Matrix<double, m, m * n> R;
  Eigen::Matrix<double, m, m> Apow;

  for (size_t i = 0; i < m; ++i) {
    Apow = Eigen::Matrix<double, m, m>::Identity();
    for (size_t j = 0; j < i; ++j) {
      Apow = Apow * A;
    }
    R.template block<m, n>(0, i * n) = Apow * B;
  }

  Eigen::FullPivLU<Eigen::Matrix<double, m, m * n>> ld(R);
  return ld.rank() == m;
}

template <size_t m, size_t n>
Eigen::Matrix<double, m, m> LQR<m, n>::solveSchur(const Eigen::Matrix<double, 2 * m, 2 * m> Z) const {
  Eigen::Matrix<double, m, m> U11, U21;
  Eigen::Matrix<double, 2 * m, 2 * m> U;

  Eigen::RealSchur<Eigen::Matrix<double, 2 * m, 2 * m>> schur(Z);

  U = schur.matrixU();

  U11 = U.template block<m, m>(0, 0);
  U21 = U.template block<m, m>(m, 0);

  return U21 * U11.inverse();
}

template class LQR<2, 1>;