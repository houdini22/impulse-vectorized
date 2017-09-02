#ifndef BUILDER_H
#define BUILDER_H

#include <string>
#include "../Network.h"
#include "../Layer/Logistic.h"
#include "../Layer/Relu.h"
#include "../../../Vendor/json.hpp"
#include "../../common.h"

using Impulse::NeuralNetwork::Network;
using Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC;
using Impulse::NeuralNetwork::Layer::TYPE_RELU;
using LogisticLayer = Impulse::NeuralNetwork::Layer::Logistic;
using ReluLayer = Impulse::NeuralNetwork::Layer::Relu;
using Impulse::T_Size;
using Impulse::NeuralNetwork::Math::rawToVector;

using json = nlohmann::json;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            class Builder {
            protected:
                Network *network;
                T_Size prevSize;
            public:
                Builder(T_Size inputSize) {
                    this->network = new Network(inputSize);
                    this->prevSize = inputSize;
                }

                void createLayer(T_Size size, T_String type) {
                    if (type == TYPE_LOGISTIC) {
                        this->network->addLayer(new LogisticLayer(size, this->prevSize));
                    } else if (type == TYPE_RELU) {
                        this->network->addLayer(new ReluLayer(size, this->prevSize));
                    }
                    this->prevSize = size;
                }

                Network *getNetwork() {
                    return this->network;
                }

                static Builder fromJSON(T_String path) {
                    std::ifstream fileStream(path);
                    json jsonFile;
                    fileStream >> jsonFile;
                    fileStream.close();

                    Builder builder((T_Size) jsonFile["inputSize"]);

                    json savedLayers = jsonFile["layers"];
                    T_Size i = 0;
                    for (auto it = savedLayers.begin(); it != savedLayers.end(); ++it) {
                        builder.createLayer(it.value()[0], it.value()[1]);
                    }

                    T_RawVector theta = jsonFile["weights"];
                    builder.getNetwork()->setRolledTheta(rawToVector(theta));

                    return builder;
                }
            };
        }
    }
}

#endif //BUILDER_H
