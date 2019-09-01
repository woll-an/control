#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "lqr.h"

class LQRUncontrollable : public ::testing::Test {
 protected:
  void SetUp() override {
    Eigen::Matrix<double, 2, 2> A;
    Eigen::Matrix<double, 2, 1> B;

    A << -2.0, 0.0, 0.0, 3.0;
    B << 0.0, 1.0;

    lqr.setState(A, B);
  }

  LQR<2, 1> lqr{};
};

class LQRControllable : public ::testing::Test {
 protected:
  void SetUp() override {
    Eigen::Matrix<double, 2, 2> A;
    Eigen::Matrix<double, 2, 1> B;

    A << 0.0, 1.1, -1.3, -0.5;
    B << 0.0, 1.0;

    lqr.setState(A, B);
  }

  LQR<2, 1> lqr{};
};

TEST_F(LQRControllable, Controllability) {
  EXPECT_TRUE(lqr.isControllable());
}

TEST_F(LQRUncontrollable, Controllability) {
  EXPECT_FALSE(lqr.isControllable());
}

TEST_F(LQRControllable, Gains) {
  Eigen::Matrix<double, 2, 1> K, Kexp;
  Eigen::Matrix<double, 2, 2> Q;
  Eigen::Matrix<double, 1, 1> R;

  Q << 2.0, 0.0, 0.0, 1.0;
  R << 1.0;

  K = lqr.computeK(Q, R);
  Kexp << 0.621, 1.117;
  EXPECT_TRUE(K.isApprox(Kexp, 0.001));
}

TEST_F(LQRUncontrollable, Gains) {
  Eigen::Matrix<double, 2, 2> Q;
  Eigen::Matrix<double, 1, 1> R;

  Q << 2.0, 0.0, 0.0, 1.0;
  R << 1.0;
  ASSERT_THROW(lqr.computeK(Q, R), std::invalid_argument);
}