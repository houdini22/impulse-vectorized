#ifndef IMPULSE_VECTORIZED_SIGMOID_H
#define IMPULSE_VECTORIZED_SIGMOID_H

#include <string>
#include "Abstract.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const std::string TYPE_SIGMOID = "sigmoid";

            class Sigmoid : public Abstract {
            protected:
            public:

                Sigmoid(unsigned int size, unsigned int prevSize) : Abstract(size, prevSize) {

                }

                Eigen::MatrixXd activation(Eigen::MatrixXd input) {
                    return input.unaryExpr([](const double x) {
                        return 1.0 / (1.0 + exp(-x));
                    });
                }

                Eigen::MatrixXd backward(Eigen::MatrixXd delta) {
                    // num examples
                    long m = delta.cols();

                    Eigen::MatrixXd result;

                    return result;
                }

                Eigen::MatrixXd derivative() {
                    return this->A.array() * (1.0 - this->A.array());
                }

                void updateParameters(double learningRate) {
                    this->W += learningRate * this->dW;
                    this->b += learningRate * this->db;
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_SIGMOID_H
