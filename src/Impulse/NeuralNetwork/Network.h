#ifndef IMPULSE_NETWORK_H_H
#define IMPULSE_NETWORK_H_H

#include <vector>
#include <iostream>
#include "Layer/Abstract.h"
#include "Math/Matrix.h"

using Matrix = Impulse::NeuralNetwork::Math::T_Matrix;
using Vector = Impulse::NeuralNetwork::Math::T_Vector;

namespace Impulse {

    namespace NeuralNetwork {

        typedef std::vector<Impulse::NeuralNetwork::Layer::Abstract *> LayerContainer;
        typedef std::vector<double> RolledTheta;

        class Network {
        protected:
            unsigned int size = 0;
            unsigned int inputSize = 0;
            LayerContainer layers;
        public:
            Network(unsigned int inputSize) {
                this->inputSize = inputSize;
            }

            void addLayer(Impulse::NeuralNetwork::Layer::Abstract *layer) {
                this->size++;
                this->layers.push_back(layer);
            }

            Matrix forward(Matrix input) {
                Matrix output = input;

                for (LayerContainer::iterator it = this->layers.begin(); it != this->layers.end(); ++it) {
                    output = (*it)->forward(output);
                }

                return output;
            }

            void backward(Matrix X, Matrix Y, Matrix predictions, double regularization) {
                long m = X.cols();
                unsigned int size = this->getSize();

                Matrix dZ = predictions.array() - Y.array();
                Matrix sigma = dZ;

                for (long i = this->layers.size() - 1; i >= 0; i--) {
                    auto layer = this->layers.at(i);

                    dZ = layer->calculateDerivative(dZ);

                    layer->calculateGradient(dZ,
                                                    (i == 0 ?
                                                     X :
                                                     this->layers.at(i - 1)->A),
                                                    regularization,
                                                    (double) m
                    );


                    dZ = this->layers.at(i)->W.transpose() * dZ;
                }
            }

            void updateParameters(double learningRate) {
                for (unsigned int layer = 0; layer < this->getSize(); layer++) {
                    this->layers.at(layer)->updateParameters(learningRate);
                }
            }

            unsigned int getInputSize() {
                return this->inputSize;
            }

            unsigned int getSize() {
                return this->size;
            }

            Impulse::NeuralNetwork::Layer::Abstract *getLayer(unsigned int key) {
                return this->layers.at(key);
            }

            Vector getRolledTheta() {
                std::vector<double> tmp;

                for (long i = 0; i < this->layers.size(); i++) {
                    auto layer = this->layers.at(i);
                    tmp.reserve(
                            (unsigned long) (layer->W.cols() * layer->W.rows()) + (layer->b.cols() * layer->b.rows()));

                    for (unsigned int j = 0; j < layer->W.rows(); j++) {
                        for (unsigned int k = 0; k < layer->W.cols(); k++) {
                            tmp.push_back(layer->W(j, k));
                        }
                    }

                    for (unsigned int j = 0; j < layer->b.rows(); j++) {
                        for (unsigned int k = 0; k < layer->b.cols(); k++) {
                            tmp.push_back(layer->b(j, k));
                        }
                    }
                }

                Vector result = Eigen::Map<Vector, Eigen::Unaligned>(tmp.data(), tmp.size());
                return result;
            }

            Vector getRolledGradient() {
                std::vector<double> tmp;

                for (unsigned long i = 0; i < this->layers.size(); i++) {
                    auto layer = this->layers.at(i);

                    for (unsigned int j = 0; j < layer->wDerivative.rows(); j++) {
                        for (unsigned int k = 0; k < layer->wDerivative.cols(); k++) {
                            tmp.push_back(layer->wDerivative(j, k));
                        }
                    }

                    for (unsigned int j = 0; j < layer->bDerivative.rows(); j++) {
                        for (unsigned int k = 0; k < layer->bDerivative.cols(); k++) {
                            tmp.push_back(layer->bDerivative(j, k));
                        }
                    }
                }

                Vector result = Eigen::Map<Vector, Eigen::Unaligned>(tmp.data(), tmp.size());
                return result;
            }

            void setRolledTheta(Vector theta) {
                unsigned long t = 0;

                for (unsigned long i = 0; i < this->layers.size(); i++) {
                    auto layer = this->layers.at(i);
                    for (unsigned int j = 0; j < layer->W.rows(); j++) {
                        for (unsigned int k = 0; k < layer->W.cols(); k++) {
                            layer->W(j, k) = theta(t++);
                        }
                    }
                    for (unsigned int j = 0; j < layer->b.rows(); j++) {
                        for (unsigned int k = 0; k < layer->b.cols(); k++) {
                            layer->b(j, k) = theta(t++);
                        }
                    }
                }
            }
        };

    }

}

#endif //IMPULSE_NETWORK_H_H
