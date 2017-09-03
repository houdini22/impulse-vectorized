#ifndef BUILDER_H
#define BUILDER_H

#include <fstream>
#include "Network.h"
#include "Layer/Logistic.h"
#include "Layer/Relu.h"
#include "../../Vendor/json.hpp"
#include "../common.h"

using namespace Impulse::NeuralNetwork::Layer;
using namespace Impulse::NeuralNetwork::Math;
using Impulse::NeuralNetwork::Network;
using Impulse::T_Size;

using json = nlohmann::json;

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
