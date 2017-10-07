#ifndef IMPULSE_NEURALNETWORK_LAYER_PURELIN_H
#define IMPULSE_NEURALNETWORK_LAYER_PURELIN_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_PURELIN = "purelin";

            class Purelin : public Abstract {
            protected:
            public:
                Purelin(T_Size size, T_Size prevSize);

                Math::T_Matrix activation() override;

                Math::T_Matrix derivative() override;

                const T_String getType() override;

                double loss(Math::T_Matrix output, Math::T_Matrix predictions) override;

                double error(T_Size m) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_PURELIN_H
