#ifndef IMPULSE_NETWORK_H_H
#define IMPULSE_NETWORK_H_H

#include <vector>
#include <iostream>
#include "Layer/Abstract.h"
#include "Math/types.h"
#include "../types.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::NeuralNetwork::Math::T_Vector;
using Impulse::T_Size;
using AbstractLayer = Impulse::NeuralNetwork::Layer::Abstract;

namespace Impulse {

    namespace NeuralNetwork {

        typedef std::vector<Impulse::NeuralNetwork::Layer::Abstract *> LayersContainer;
        typedef std::vector<double> RolledTheta;

        class Network {
        protected:
            T_Size size = 0;
            T_Size inputSize = 0;
            LayersContainer layers;
        public:
            Network(T_Size inputSize) {
                this->inputSize = inputSize;
            }

            void addLayer(Impulse::NeuralNetwork::Layer::Abstract *layer) {
                this->size++;
                this->layers.push_back(layer);
            }

            T_Matrix forward(T_Matrix input) {
                T_Matrix output = input;

                for (LayersContainer::iterator it = this->layers.begin(); it != this->layers.end(); ++it) {
                    output = (*it)->forward(output);
                }

                return output;
            }

            void backward(T_Matrix X, T_Matrix Y, T_Matrix predictions, double regularization) {
                long m = X.cols();
                T_Size size = this->getSize();

                T_Matrix sigma = predictions.array() - Y.array();

                for (long i = this->layers.size() - 1; i >= 0; i--) {
                    auto layer = this->layers.at(i);

                    T_Matrix delta = sigma * (i == 0 ? X : this->layers.at(i - 1)->A).transpose().conjugate();
                    layer->gW = delta.array() / m + (regularization / m * layer->W.array());
                    layer->gb = sigma.rowwise().sum() / m;

                    if (i > 0) {
                        auto prevLayer = this->layers.at(i - 1);

                        T_Matrix tmp1 = layer->W.transpose() * sigma;
                        T_Matrix tmp2 = prevLayer->derivative();

                        sigma = tmp1.array() * tmp2.array();
                    }
                }
            }

            T_Size getInputSize() {
                return this->inputSize;
            }

            T_Size getSize() {
                return this->size;
            }

            AbstractLayer *getLayer(T_Size key) {
                return this->layers.at(key);
            }

            T_Vector getRolledTheta() {
                std::vector<double> tmp;

                for (long i = 0; i < this->layers.size(); i++) {
                    auto layer = this->layers.at(i);
                    tmp.reserve(
                            (unsigned long) (layer->W.cols() * layer->W.rows()) + (layer->b.cols() * layer->b.rows()));

                    for (T_Size j = 0; j < layer->W.rows(); j++) {
                        for (T_Size k = 0; k < layer->W.cols(); k++) {
                            tmp.push_back(layer->W(j, k));
                        }
                    }

                    for (T_Size j = 0; j < layer->b.rows(); j++) {
                        for (T_Size k = 0; k < layer->b.cols(); k++) {
                            tmp.push_back(layer->b(j, k));
                        }
                    }
                }

                T_Vector result = Eigen::Map<T_Vector, Eigen::Unaligned>(tmp.data(), tmp.size());
                return result;
            }

            T_Vector getRolledGradient() {
                std::vector<double> tmp;

                for (unsigned long i = 0; i < this->layers.size(); i++) {
                    auto layer = this->layers.at(i);

                    for (T_Size j = 0; j < layer->gW.rows(); j++) {
                        for (T_Size k = 0; k < layer->gW.cols(); k++) {
                            tmp.push_back(layer->gW(j, k));
                        }
                    }

                    for (T_Size j = 0; j < layer->gb.rows(); j++) {
                        for (T_Size k = 0; k < layer->gb.cols(); k++) {
                            tmp.push_back(layer->gb(j, k));
                        }
                    }
                }

                T_Vector result = Eigen::Map<T_Vector, Eigen::Unaligned>(tmp.data(), tmp.size());
                return result;
            }

            void setRolledTheta(T_Vector theta) {
                unsigned long t = 0;

                for (unsigned long i = 0; i < this->layers.size(); i++) {
                    auto layer = this->layers.at(i);
                    for (T_Size j = 0; j < layer->W.rows(); j++) {
                        for (T_Size k = 0; k < layer->W.cols(); k++) {
                            layer->W(j, k) = theta(t++);
                        }
                    }
                    for (T_Size j = 0; j < layer->b.rows(); j++) {
                        for (T_Size k = 0; k < layer->b.cols(); k++) {
                            layer->b(j, k) = theta(t++);
                        }
                    }
                }
            }
        };

    }

}

#endif //IMPULSE_NETWORK_H_H
