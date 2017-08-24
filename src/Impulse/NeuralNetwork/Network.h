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
                /*Eigen::MatrixXd delta = (Y.array() - predictions.array());

                for (long i = this->layers.size() - 1; i >= 0; i--) {
                    delta = this->layers.at(i)->backward(delta);
                    if (i > 0) {
                        delta.array() *= this->layers.at(i - 1)->derivative().array();
                    }
                }*/
                long m = Y.cols();

                Eigen::MatrixXd E = (Y.array() - predictions.array());
                Eigen::MatrixXd dZ = E.array() * (this->layers.at(1)->derivative().array());
                Eigen::MatrixXd dH = (this->layers.at(1)->W.transpose() * dZ).array() * (this->layers.at(0)->derivative().array());

                /*std::cout << "DZ" << std::endl << dZ << std::endl << std::endl;
                std::cout << "DH" << std::endl << dH << std::endl << std::endl;
                std::cout << "db2" << std::endl << dZ.rowwise().sum() << std::endl << std::endl;
                std::cout << "db1" << std::endl << dH.rowwise().sum() << std::endl << std::endl;
                std::cout << "dW2" << std::endl << dZ * this->layers.at(0)->A.transpose() << std::endl << std::endl;
                std::cout << "dW1" << std::endl << dH * X.transpose() << std::endl << std::endl;*/

                this->layers.at(0)->W += dH * X.transpose();
                this->layers.at(1)->W += dZ * this->layers.at(0)->A.transpose();
                this->layers.at(0)->b += dH.rowwise().sum();
                this->layers.at(1)->b += dZ.rowwise().sum();

                /*std::cout << dZ << std::endl << std::endl;
                std::cout << dH << std::endl << std::endl;

                dZ = (1.0 / (double) m) * (dZ.rowwise().sum()).array();
                dH = (1.0 / (double) m) * (dH.rowwise().sum()).array();

                std::cout << dZ << std::endl << std::endl;
                std::cout << dH << std::endl << std::endl;*/
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
