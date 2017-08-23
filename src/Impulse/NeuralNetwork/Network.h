#ifndef IMPULSE_NETWORK_H_H
#define IMPULSE_NETWORK_H_H

#include <vector>
#include <iostream>
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

            void backward(Eigen::MatrixXd X, Eigen::MatrixXd Y, Eigen::MatrixXd predictions) {
                Eigen::MatrixXd A = Y.array() - predictions.array();
                for (long i = this->layers.size() - 1; i >= 0; i--) {
                    A = this->layers.at(i)->backward(A, i == 0 ? X : this->layers.at(i - 1)->getA());
                }
            }

            void updateParameters(double learningRate) {
                for (unsigned int layer = 0; layer < this->getSize(); layer++) {
                    this->layers.at(layer)->updateParameters(learningRate);
                }
            }
        };

    }

}

#endif //IMPULSE_NETWORK_H_H
