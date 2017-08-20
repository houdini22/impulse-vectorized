#ifndef IMPULSE_VECTORIZED_BUILDER_H
#define IMPULSE_VECTORIZED_BUILDER_H

#include <string>
#include "../Network.h"
#include "../Layer/Sigmoid.h"
#include "../Layer/Input.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            class Builder {
            protected:
                Impulse::NeuralNetwork::Network *network;
                unsigned int prevSize;
            public:
                Builder() {
                    this->network = new Impulse::NeuralNetwork::Network();
                }

                void createLayer(unsigned int size, std::string type) {
                    if (this->network->getSize() == 0) {
                        this->network->addLayer(new Impulse::NeuralNetwork::Layer::Input(size));
                        this->prevSize = size;
                    }
                    if (type == "sigmoid") {
                        this->network->addLayer(new Impulse::NeuralNetwork::Layer::Sigmoid(size, this->prevSize));
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
