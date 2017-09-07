#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include <string>
#include "Abstract.h"
#include "../Math/common.h"
#include "../../common.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::T_Size;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_SOFTMAX = "softmax";

            class Softmax : public Abstract {
            protected:
            public:

                Softmax(T_Size size, T_Size prevSize) : Abstract(size, prevSize) {}

                T_Matrix activation();

                T_Matrix derivative();

                const T_String getType();

                double loss(T_Matrix output, T_Matrix predictions);
            };
        }
    }
}

#endif //SOFTMAX_LAYER_H
