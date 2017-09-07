#ifndef RELU_LAYER_H
#define RELU_LAYER_H

#include <string>
#include "Abstract.h"
#include "../Math/common.h"
#include "../../common.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::T_Size;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_RELU = "relu";

            class Relu : public Abstract {
            protected:
            public:

                Relu(T_Size size, T_Size prevSize) : Abstract(size, prevSize) {}

                T_Matrix activation();

                T_Matrix derivative();

                const T_String getType();

                double loss(T_Matrix output, T_Matrix predictions);
            };
        }
    }
}

#endif //RELU_LAYER_H
