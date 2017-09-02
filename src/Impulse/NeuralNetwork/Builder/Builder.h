#ifndef IMPULSE_VECTORIZED_BUILDER_H
#define IMPULSE_VECTORIZED_BUILDER_H

#include <string>
#include "../Network.h"
#include "../Layer/Logistic.h"
#include "../Layer/Relu.h"
#include "../../../Vendor/json.hpp"
#include "../../types.h"

using json = nlohmann::json;

using Impulse::NeuralNetwork::Network;
using Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC;
using Impulse::NeuralNetwork::Layer::TYPE_RELU;
using LogisticLayer = Impulse::NeuralNetwork::Layer::Logistic;
using ReluLayer = Impulse::NeuralNetwork::Layer::Relu;
using Impulse::T_Size;

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

                void createLayer(T_Size size, std::string type) {
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

                static Builder fromJSON(std::string path) {
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

                    std::vector<double> theta = jsonFile["weights"];

                    builder.getNetwork()->setRolledTheta(Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(theta.data(), theta.size()));

                    return builder;
                }
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_BUILDER_H
