#ifndef IMPULSE_VECTORIZED_ACTIVATE_FUNCTION_RELU_H
#define IMPULSE_VECTORIZED_ACTIVATE_FUNCTION_RELU_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace ActivationFunction {

                inline Math::T_Matrix reluActivation(Math::T_Matrix m) {
                    return Math::Matrix::forEach(m, [](const double &x) {
                        return std::max(0.0, x);
                    });
                }

                inline Math::T_Matrix logisticActivation(Math::T_Matrix m) {
                    return Math::Matrix::forEach(m, [](const double &x) {
                        return 1.0 / (1.0 + exp(-x));
                    });
                }
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_ACTIVATE_FUNCTION_RELU_H
