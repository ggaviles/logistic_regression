#include <cmath>
#include <vector>


//Define sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

double predictOne(const std::vector<double> &x, const std::vector<double> &w) {
    double z = w[0];
    for (size_t i = 0; i < x.size(); i++) {
        z += w[i+1] * x[i];
    }

    double prediction = sigmoid(z);

    return prediction;

}

double computeCost(
    const std::vector<std::vector<double>> &X,
    const std::vector<int> &y,
    const std::vector<double> &w
) {
    double cost = 0.0;
    int N = X.size();
    for (int i = 0; i < N; ++i) {
        double prediction = predictOne(X[i], w);
        cost += y[i] * std::log(prediction + 1e-9) + (1 - y[i]) * std::log(1 - prediction + 1e-9);
    }
    return -cost / N;
}
