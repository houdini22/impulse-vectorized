#ifndef IMPULSE_NEURALNETWORK_LAYER_SOFTMAX_H
#define IMPULSE_NEURALNETWORK_LAYER_SOFTMAX_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_SOFTMAX = "softmax";

            class Softmax : public Abstract {
            protected:
            public:

                Softmax(T_Size size, T_Size prevSize);

                Math::T_Matrix activation();

                Math::T_Matrix derivative();

                const T_String getType();

                double loss(Math::T_Matrix output, Math::T_Matrix predictions);
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_SOFTMAX_H
