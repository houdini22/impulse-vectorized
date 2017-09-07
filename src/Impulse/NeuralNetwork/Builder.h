#ifndef BUILDER_H
#define BUILDER_H

#include "include.h"

using namespace Impulse::NeuralNetwork;
using Impulse::NeuralNetwork::Network;

namespace Impulse {

    namespace NeuralNetwork {

        class Builder {
        protected:
            Network *network;
            T_Size prevSize;
        public:
            Builder(T_Size inputSize);

            void createLayer(T_Size size, T_String type);

            Network *getNetwork();

            static Builder fromJSON(T_String path);
        };
    }
}

#endif //BUILDER_H
