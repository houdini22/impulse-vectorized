#ifndef IMPULSE_VECTORIZED_ACTIVATE_FUNCTION_RELU_H
#define IMPULSE_VECTORIZED_ACTIVATE_FUNCTION_RELU_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace ActivationFunction {

                inline Math::T_Matrix relu(Math::T_Matrix m) {
                    return Math::Matrix::forEach(m, [](const double &x) {
                        return std::max(0.0, x);
                    });
                }

                inline Math::T_Matrix logistic(Math::T_Matrix m) {
                    return Math::Matrix::forEach(m, [](const double &x) {
                        return 1.0 / (1.0 + exp(-x));
                    });
                }

                inline Math::T_Matrix softmax(Math::T_Matrix m) {
                    Math::T_Matrix t = Math::Matrix::exp(m);
                    Math::T_Matrix divider = Math::Matrix::replicateRows(Math::Matrix::colwiseSum(t), Math::Matrix::rows(t));
                    Math::T_Matrix result = Math::Matrix::divide(t, divider);
                    return result;
                }
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_ACTIVATE_FUNCTION_RELU_H
