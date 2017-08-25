#ifndef IMPULSE_VECTORIZED_BUILDER_H
#define IMPULSE_VECTORIZED_BUILDER_H

#include <string>
#include "../Network.h"
#include "../Layer/Logistic.h"
#include "../Layer/Relu.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            class Builder {
            protected:
                Impulse::NeuralNetwork::Network *network;
                unsigned int prevSize;
            public:
                Builder(unsigned int inputSize) {
                    this->network = new Impulse::NeuralNetwork::Network();
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
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_BUILDER_H
