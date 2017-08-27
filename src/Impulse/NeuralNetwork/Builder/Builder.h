#ifndef IMPULSE_VECTORIZED_BUILDER_H
#define IMPULSE_VECTORIZED_BUILDER_H

#include <string>
#include "../Network.h"
#include "../Layer/Logistic.h"
#include "../Layer/Relu.h"
#include "../../../Vendor/json.hpp"

using json = nlohmann::json;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            class Builder {
            protected:
                Impulse::NeuralNetwork::Network *network;
                unsigned int prevSize;
            public:
                Builder(unsigned int inputSize) {
                    this->network = new Impulse::NeuralNetwork::Network(inputSize);
                    this->prevSize = inputSize;
                }

                void createLayer(unsigned int size, std::string type) {
                    if (type == Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC) {
                        this->network->addLayer(new Impulse::NeuralNetwork::Layer::Logistic(size, this->prevSize));
                    } else if (type == Impulse::NeuralNetwork::Layer::TYPE_RELU) {
                        this->network->addLayer(new Impulse::NeuralNetwork::Layer::Relu(size, this->prevSize));
                    }
                    this->prevSize = size;
                }

                Impulse::NeuralNetwork::Network *getNetwork() {
                    return this->network;
                }

                static Builder fromJSON(std::string path) {
                    /*std::ifstream fileStream(path);
                    json jsonFile;
                    fileStream >> jsonFile;
                    fileStream.close();

                    Builder builder((unsigned int) jsonFile["inputSize"]);

                    json savedLayers = jsonFile["layers"];
                    unsigned int i = 0;
                    for (auto it = savedLayers.begin(); it != savedLayers.end(); ++it) {
                        builder.createLayer(it.value()[0], it.value()[1]);
                    }

                    std::vector<double> theta = jsonFile["weights"];

                    builder.getNetwork()->setRolledTheta(Eigen::Map<Eigen::VectorXd, Eigen::Unaligned>(theta.data(), theta.size()));

                    return builder;*/
                }
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_BUILDER_H
