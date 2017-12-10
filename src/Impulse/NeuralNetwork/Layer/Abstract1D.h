#ifndef IMPULSE_NEURALNETWORK_LAYER_2D_H
#define IMPULSE_NEURALNETWORK_LAYER_2D_H

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract2D : public Abstract {
            public:
                Abstract2D();

                void configure() override;

                bool is2d() override;

                bool is3d() override;

                void transition(const Layer::LayerPointer &prevLayer) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_2D_H
