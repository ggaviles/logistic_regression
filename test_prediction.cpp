#include <iostream>
#include <cassert>
#include <cmath>
#include "LogisticRegression.h"

bool approxEqual(double a, double b, double epsilon = 1e-6) {
    return std::fabs(a - b) < epsilon;
}

std::vector<double> x = {2.0, 3.0};
std::vector<double> w = {0.5, 1.0, -0.5};

int main() {
    assert(approxEqual(predictOne(x, w), 0.731, 1e-2));

    std::cout << "Prediction: " << predictOne(x, w) << std::endl;

    std::cout << "All tests passed!" << std::endl;
    return 0;
}