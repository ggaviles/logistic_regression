#include <iostream>
#include <cassert>
#include <cmath>
#include "LogisticRegression.h"

bool approxEqual(double a, double b, double epsilon = 1e-6) {
    return std::fabs(a - b) < epsilon;
}

int main() {
    assert(approxEqual(sigmoid(0.0), 0.5));
    assert(approxEqual(sigmoid(10.0), 1.0 / (1.0 + std::exp(-10.0))));
    assert(approxEqual(sigmoid(-10.0), 1.0 / (1.0 + std::exp(10.0))));

    std::cout <<"All tests passed!" << std::endl;
    return 0;
}