#ifndef IMPULSE_NEURALNETWORK_LAYER_POOL_H
#define IMPULSE_NEURALNETWORK_LAYER_POOL_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_POOL = "pool";

            class Pool : public Abstract {
            protected:
                T_Size width = 0;
                T_Size height = 0;
                T_Size depth = 0;
                T_Size filterSize = 2;
                T_Size stride = 2;
            public:
                Pool();

                void configure() override;

                void setFilterSize(T_Size value);

                void setStride(T_Size value);

                Math::T_Matrix forward(const Math::T_Matrix &input) override;

                Math::T_Matrix activation() override;

                Math::T_Matrix derivative() override;

                const T_String getType() override;

                double loss(Math::T_Matrix output, Math::T_Matrix predictions) override;

                double error(T_Size m) override;

                void transition(Layer::LayerPointer prevLayer) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_POOL_H