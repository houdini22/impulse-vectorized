#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H

#include <string>
#include "Abstract.h"
#include "../Math/common.h"
#include "../../common.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::T_Size;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_LOGISTIC = "logistic";

            class Logistic : public Abstract {
            protected:
            public:

                Logistic(T_Size size, T_Size prevSize) : Abstract(size, prevSize) {

                }

                T_Matrix activation() {
                    return this->Z.unaryExpr([](const double x) {
                        return 1.0 / (1.0 + exp(-x));
                    });
                }

                T_Matrix derivative() {
                    return this->A.array() * (1.0 - this->A.array());
                }

                const T_String getType() {
                    return TYPE_LOGISTIC;
                }
            };
        }
    }
}

#endif //LOGISTIC_LAYER_H
