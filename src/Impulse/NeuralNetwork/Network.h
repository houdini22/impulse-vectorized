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

                std::cout << output.rows() << "," << output.cols() << std::endl;

                assert(output.cols() == input.cols());
                //assert(output.rows() == input.rows());

                return output;
            }

            void backward(Eigen::MatrixXd predictions, Eigen::MatrixXd Y) {
                Eigen::MatrixXd dA =
                        (Y.array() / predictions.array()) +
                        ((Y.unaryExpr([](const double x) { return 1.0 - x; }).array()))
                        /
                        (
                                predictions.unaryExpr([](const double x) { return 1.0 - x; }).array()
                        );
                for (unsigned int layer = this->getSize() - 1; layer > 0; layer--) {
                    dA = this->layers.at(layer)->backward(dA);
                }
            }

            void updateParameters(double learningRate) {
                for (unsigned int layer = this->getSize() - 1; layer > 0; layer--) {
                    this->layers.at(layer)->updateParameters(learningRate);
                }
            }
        };

    }

}

#endif //IMPULSE_NETWORK_H_H
