#include <stdexcept>

#include "lqr.h"

template <size_t m, size_t n>
LQR<m, n>::LQR(Eigen::Matrix<double, m, m>& A, Eigen::Matrix<double, m, n>& B) : A(A), B(B) {}

template <size_t m, size_t n>
Eigen::Matrix<double, m, n> LQR<m, n>::computeK(Eigen::Matrix<double, m, m>& Q,
                                                Eigen::Matrix<double, n, n>& R) {
  if (!isControllable()) {
    throw std::invalid_argument("system is not controllable");
  }

  Eigen::Matrix<double, m, n> K;
  Eigen::Matrix<double, m, m> P;
  Eigen::Matrix<double, m, m> U11, U21;
  Eigen::Matrix<double, 2 * m, 2 * m> Z, U;

  Z << A, -B * R.inverse() * B.transpose(), -Q, -A.transpose();
  Eigen::RealSchur<Eigen::Matrix<double, 2 * m, 2 * m>> schur(Z);
  U = schur.matrixU();
  U11 = U.template block<m, m>(0, 0);
  U21 = U.template block<m, m>(m, 0);
  P = U21 * U11.inverse();
  K = R.inverse() * (B.transpose() * P);

  return K;
};

template <size_t m, size_t n>
bool LQR<m, n>::isControllable() {
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

template class LQR<2, 1>;