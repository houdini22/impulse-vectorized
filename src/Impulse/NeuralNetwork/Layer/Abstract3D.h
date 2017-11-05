#ifndef IMPULSE_NEURALNETWORK_LAYER_3D_H
#define IMPULSE_NEURALNETWORK_LAYER_3D_H

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract3D : public Abstract {
            public:
                Abstract3D();

                bool is2d() override;

                bool is3d() override;

                void transition(const Layer::LayerPointer &prevLayer) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_3D_H
