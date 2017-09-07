#ifndef SOFTMAX_LAYER_H
#define SOFTMAX_LAYER_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_SOFTMAX = "softmax";

            class Softmax : public Abstract {
            protected:
            public:

                Softmax(T_Size size, T_Size prevSize) : Abstract(size, prevSize) {}

                Math::T_Matrix activation();

                Math::T_Matrix derivative();

                const T_String getType();

                double loss(Math::T_Matrix output, Math::T_Matrix predictions);
            };
        }
    }
}

#endif //SOFTMAX_LAYER_H
