#ifndef IMPULSE_VECTORIZED_RELU_H
#define IMPULSE_VECTORIZED_RELU_H

#include <string>
#include "Abstract.h"
#include "../Math/types.h"
#include "../../types.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::T_Size;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const std::string TYPE_RELU = "relu";

            class Relu : public Abstract {
            protected:
            public:

                Relu(T_Size size, T_Size prevSize) : Abstract(size, prevSize) {

                }

                T_Matrix activation() {
                    return this->Z.unaryExpr([](const double x) {
                        if (x < 0.0) {
                            return 0.0;
                        }
                        return x;
                    });
                }

                T_Matrix derivative() {
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
