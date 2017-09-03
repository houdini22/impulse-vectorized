#include "Network.h"

namespace Impulse {

    namespace NeuralNetwork {

        Network::Network(T_Size inputSize) {
            this->inputSize = inputSize;
        }

        void Network::addLayer(AbstractLayer *layer) {
            this->size++;
            this->layers.push_back(layer);
        }

        T_Matrix Network::forward(T_Matrix input) {
            T_Matrix output = input;

            for (LayersContainer::iterator it = this->layers.begin(); it != this->layers.end(); ++it) {
                output = (*it)->forward(output);
            }

            return output;
        }

        void Network::backward(T_Matrix X, T_Matrix Y, T_Matrix predictions, double regularization) {
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

        T_Size Network::getInputSize() {
            return this->inputSize;
        }

        T_Size Network::getSize() {
            return this->size;
        }

        AbstractLayer *Network::getLayer(T_Size key) {
            return this->layers.at(key);
        }

        T_Vector Network::getRolledTheta() {
            T_RawVector tmp;

            for (T_Size i = 0; i < this->getSize(); i++) {
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

            T_Vector result = rawToVector(tmp);
            return result;
        }

        T_Vector Network::getRolledGradient() {
            T_RawVector tmp;

            for (T_Size i = 0; i < this->getSize(); i++) {
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

            T_Vector result = rawToVector(tmp);
            return result;
        }

        void Network::setRolledTheta(T_Vector theta) {
            unsigned long t = 0;

            for (T_Size i = 0; i < this->getSize(); i++) {
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
    }
}
