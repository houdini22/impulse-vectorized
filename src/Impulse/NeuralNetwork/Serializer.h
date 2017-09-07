#ifndef IMPULSE_NEURALNETWORK_SERIALIZER_H
#define IMPULSE_NEURALNETWORK_SERIALIZER_H

#include "include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        class Serializer {
        protected:
            Impulse::NeuralNetwork::Network *network;
        public:
            Serializer(Impulse::NeuralNetwork::Network *net);

            void toJSON(T_String path);
        };
    }
}

#endif //IMPULSE_NEURALNETWORK_SERIALIZER_H