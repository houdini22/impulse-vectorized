#ifndef IMPULSE_NEURALNETWORK_BUILDER_H
#define IMPULSE_NEURALNETWORK_BUILDER_H

#include "include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        class Builder {
        protected:
            Network network;
            T_Size prevSize;
        public:
            Builder(T_Size inputSize);

            void createLayer(T_Size size, T_String type);

            Network &getNetwork();

            static Builder fromJSON(T_String path);
        };
    }
}

#endif //IMPULSE_NEURALNETWORK_BUILDER_H
