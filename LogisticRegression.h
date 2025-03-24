#include <vector>
#ifndef LOGISTIC_REGRESSION_H
#define LOGISTIC_REGRESSION_H

double sigmoid(double z);

double predictOne(const std::vector<double> &x, const std::vector<double> &w);

#endif