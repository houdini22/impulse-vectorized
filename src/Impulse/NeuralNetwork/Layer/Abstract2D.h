#ifndef IMPULSE_NEURALNETWORK_LAYER_3D_H
#define IMPULSE_NEURALNETWORK_LAYER_3D_H

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract2D : public Abstract {
            public:
                Abstract2D();

                bool is1D() override;

                bool is2D() override;

                void transition(Layer::LayerPointer prevLayer) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_3D_H
