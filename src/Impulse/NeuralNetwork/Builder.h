#ifndef IMPULSE_NEURALNETWORK_BUILDER_H
#define IMPULSE_NEURALNETWORK_BUILDER_H

#include "include.h"
#include "Layer/Logistic.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        class Builder {
        protected:
            Network network;
            T_Size prevSize;
        public:
            explicit Builder(T_Size inputSize);

            template<typename LAYER_TYPE>
            void createLayer(T_Size size, std::function<void(LAYER_TYPE *)> callback);

            template<typename LAYER_TYPE>
            void createLayer(T_Size size);

            template<typename LAYER_TYPE>
            void createLayer(std::function<void(LAYER_TYPE *)> callback);

            Network &getNetwork();

            static Builder fromJSON(T_String path);
        };
    }
}

#endif //IMPULSE_NEURALNETWORK_BUILDER_H
