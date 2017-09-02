#ifndef IMPULSE_VECTORIZED_SIGMOID_H
#define IMPULSE_VECTORIZED_SIGMOID_H

#include <string>
#include "Abstract.h"
#include "../Math/Matrix.h"

using Matrix = Impulse::NeuralNetwork::Math::T_Matrix;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const std::string TYPE_LOGISTIC = "logistic";

            class Logistic : public Abstract {
            protected:
            public:

                Logistic(unsigned int size, unsigned int prevSize) : Abstract(size, prevSize) {

                }

                Matrix activation(Matrix input) {
                    return input.unaryExpr([](const double x) {
                        return 1.0 / (1.0 + exp(-x));
                    });
                }

                Matrix derivative() {
                    return this->A.array() * (1.0 - this->A.array());
                }

                const std::string getType() {
                    return TYPE_LOGISTIC;
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_SIGMOID_H
