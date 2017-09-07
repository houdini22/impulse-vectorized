#ifndef NETWORK_SERIALIZER_H
#define NETWORK_SERIALIZER_H

#include <fstream>
#include <string>
#include <iostream>

#include "Network.h"
#include "../../Vendor/json.hpp"

using json = nlohmann::json;

using Impulse::NeuralNetwork::Math::T_Vector;
using Impulse::NeuralNetwork::Math::vectorToRaw;
using Impulse::NeuralNetwork::T_Size;

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

#endif
