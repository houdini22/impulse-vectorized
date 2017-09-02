#ifndef IMPULSE_VECTORIZED_RELU_H
#define IMPULSE_VECTORIZED_RELU_H

#include <string>
#include "Abstract.h"
#include "../Math/Matrix.h"

using Matrix = Impulse::NeuralNetwork::Math::T_Matrix;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const std::string TYPE_RELU = "relu";

            class Relu : public Abstract {
            protected:
            public:

                Relu(unsigned int size, unsigned int prevSize) : Abstract(size, prevSize) {

                }

                Matrix activation(Matrix input) {
                    return input.unaryExpr([](const double x) {
                        if (x < 0.0) {
                            return 0.0;
                        }
                        return x;
                    });
                }

                Matrix derivative() {
                    return this->A.unaryExpr([](const double x) {
                        if (x < 0.0) {
                            return 0.0;
                        }
                        return 1.0;
                    });
                }

                const std::string getType() {
                    return TYPE_RELU;
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_RELU_H
