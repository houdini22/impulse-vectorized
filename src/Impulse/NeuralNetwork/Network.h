#ifndef IMPULSE_NETWORK_H_H
#define IMPULSE_NETWORK_H_H

#include <vector>
#include "Layer/Abstract.h"

namespace Impulse {

    namespace NeuralNetwork {

        typedef std::vector<Impulse::NeuralNetwork::Layer::Abstract *> LayerContainer;

        class Network {
        protected:
            unsigned int size = 0;
            LayerContainer layers;
        public:
            unsigned int getSize() {
                return this->size;
            }

            void addLayer(Impulse::NeuralNetwork::Layer::Abstract *layer) {
                this->size++;
                this->layers.push_back(layer);
            }

            Eigen::MatrixXd forward(Eigen::MatrixXd input) {
                Eigen::MatrixXd output = input;
                for (LayerContainer::iterator it = this->layers.begin(); it != this->layers.end(); ++it) {
                    output = (*it)->forward(output);
                }
                return output;
            }
        };

    }

}

#endif //IMPULSE_NETWORK_H_H
