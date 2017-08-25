#ifndef IMPULSE_VECTORIZED_SIGMOID_H
#define IMPULSE_VECTORIZED_SIGMOID_H

#include <string>
#include "Abstract.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const std::string TYPE_LOGISTIC = "logistic";

            class Logistic : public Abstract {
            protected:
            public:

                Logistic(unsigned int size, unsigned int prevSize) : Abstract(size, prevSize) {

                }

                Eigen::MatrixXd activation(Eigen::MatrixXd input) {
                    return input.unaryExpr([](const double x) {
                        return 1.0 / (1.0 + exp(-x));
                    });
                }

                Eigen::MatrixXd derivative() {
                    return this->A.array() * (1.0 - this->A.array());
                }

                std::string getType() {
                    return TYPE_LOGISTIC;
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_SIGMOID_H
