#include <cmath>
#include <vector>


// Define sigmoid function
double sigmoid(double z) {
    return 1.0 / (1.0 + exp(-z));
}

// Compute single prediction
double predictOne(const std::vector<double> &x, const std::vector<double> &w) {
    double z = w[0];
    for (size_t i = 0; i < x.size(); i++) {
        z += w[i+1] * x[i];
    }

    double prediction = sigmoid(z);

    return prediction;

}

// Compute cross-entropy loss
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

void gradientDescent(
    std::vector<double> &w, //
    const std::vector<std::vector<double>> &X, // Feature vector
    const std::vector<int> &y, // 
    double alpha, // learning rate
    int epochs // iteration over all data
) {
    int N = X.size(); // N training examples
    int d = X[0].size(); // number of features
    for (int e = 0; e < epochs; ++e) {

        // Initialize an array <grad> of size (d+1) to 0 at the start of each epoch
        std::vector<double> grad(d+1, 0.0);

        // Loop over each training example i = 0 to N-1
        for (int i = 0; i < N; ++i) {
            double prediction = predictOne(X[i], w);
            double error = prediction - y[i];

            // Add partial derivatives for the bias
            grad[0] += error;

            // Add partial derivatives for each weight j = 1, ..., d
            for (int j = 0; j < d; ++j) {
                grad[j+1] += error * X[i][j];
            }
        }

        for (int j = 0; j < d+1; ++j) {
            grad[j] /= N; // Average each accumulated gradient by dividing by N
            w[j] -= alpha * grad[j]; // Update the weights
        }

        if (e % 100 == 0) {
            double c = computeCost(X, y, W);
            std::cout << "Epoch" << e << " Cost: " << c << std::endl;
        }
    }
}