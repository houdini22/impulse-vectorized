#ifndef IMPULSE_VECTORIZED_DERIVATIVE_H
#define IMPULSE_VECTORIZED_DERIVATIVE_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace Derivative {

                inline Math::T_Matrix relu(Math::T_Matrix m) {
                    return Math::Matrix::forEach(m, [](const double &x) {
                        if (x < 0.0) {
                            return 0.0;
                        }
                        return 1.0;
                    });
                }

                inline Math::T_Matrix logistic(Math::T_Matrix m) {
                    return Math::Matrix::elementWiseMultiply(m, Math::Matrix::forEach(m, [](const double &x) {
                        return 1.0 - x;
                    }));
                }
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_DERIVATIVE_H
