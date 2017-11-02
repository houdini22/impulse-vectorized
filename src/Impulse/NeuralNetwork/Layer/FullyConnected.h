#ifndef IMPULSE_NEURALNETWORK_LAYER_FULLYCONNECTED_H
#define IMPULSE_NEURALNETWORK_LAYER_FULLYCONNECTED_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_FULLYCONNECTED = "fully-connected";

            class FullyConnected : public Logistic {
            protected:
            public:
                FullyConnected();

                void transition(const Layer::LayerPointer &prevLayer) override;

                const T_String getType() override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_FULLYCONNECTED_H
