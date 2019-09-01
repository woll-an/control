#include <gtest/gtest.h>
#include <Eigen/Dense>

#include "lqr.h"

class Uncontrollable : public ::testing::Test {
 protected:
  void SetUp() override {
    A << -2.0, 0.0, 0.0, 3.0;
    B << 0.0, 1.0;
  }

  Eigen::Matrix<double, 2, 2> A;
  Eigen::Matrix<double, 2, 1> B;

  LQR<2, 1> lqr{A, B};
};

class Controllable : public ::testing::Test {
 protected:
  void SetUp() override {
    A << 0.0, 1.1, -1.3, -0.5;
    B << 0.0, 1.0;
  }

  Eigen::Matrix<double, 2, 2> A;
  Eigen::Matrix<double, 2, 1> B;

  LQR<2, 1> lqr{A, B};
};

TEST_F(Controllable, LQRControllability) {
  EXPECT_TRUE(lqr.isControllable());
}

TEST_F(Uncontrollable, LQRControllability) {
  EXPECT_FALSE(lqr.isControllable());
}

TEST_F(Controllable, LQRGains) {
  Eigen::Matrix<double, 2, 1> K, Kexp;
  Eigen::Matrix<double, 2, 2> Q;
  Eigen::Matrix<double, 1, 1> R;

  Q << 2.0, 0.0, 0.0, 1.0;
  R << 1.0;

  K = lqr.computeK(Q, R);
  Kexp << 0.621, 1.117;
  EXPECT_TRUE(K.isApprox(Kexp, 0.001));
}

TEST_F(Uncontrollable, LQRGains) {
  Eigen::Matrix<double, 2, 2> Q;
  Eigen::Matrix<double, 1, 1> R;

  Q << 2.0, 0.0, 0.0, 1.0;
  R << 1.0;
  ASSERT_THROW(lqr.computeK(Q, R), std::invalid_argument);
}