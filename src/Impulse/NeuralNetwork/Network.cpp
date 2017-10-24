#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        Network::Network(T_Size inputSize) {
            this->inputSize = inputSize;
        }

        void Network::addLayer(Layer::LayerPointer layer) {
            this->size++;
            this->layers.push_back(layer);
        }

        Math::T_Matrix Network::forward(Math::T_Matrix input) {
            Math::T_Matrix output = input;
            Layer::LayerPointer prevLayer = nullptr;

            for (auto &layer : this->layers) {
                layer->transition(prevLayer);
                output = layer->forward(output);
                prevLayer = layer;
            }

            return output;
        }

        void Network::backward(Math::T_Matrix X, Math::T_Matrix Y, Math::T_Matrix predictions, double regularization) {
            long m = X.cols();
            T_Size size = this->getSize();

            Math::T_Matrix sigma = predictions.array() - Y.array();

            for (long i = this->layers.size() - 1; i >= 0; i--) {
                auto layer = this->layers.at(static_cast<unsigned long>(i));

                Math::T_Matrix delta = sigma * (i == 0 ? X : this->layers.at(
                        static_cast<unsigned long>(i - 1))->A).transpose().conjugate();

                layer->gW = delta.array() / m + (regularization / m * layer->W.array());
                layer->gb = sigma.rowwise().sum() / m;

                if (i > 0) {
                    auto prevLayer = this->layers.at(static_cast<unsigned long>(i - 1));

                    Math::T_Matrix tmp1 = layer->W.transpose() * sigma;
                    Math::T_Matrix tmp2 = prevLayer->derivative();

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

        Layer::Abstract *Network::getLayer(T_Size key) {
            return this->layers.at(key).get();
        }

        Math::T_Vector Network::getRolledTheta() {
            Math::T_RawVector tmp;

            for (T_Size i = 0; i < this->getSize(); i++) {
                auto layer = this->getLayer(i);
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

            Math::T_Vector result = Math::rawToVector(tmp);
            return result;
        }

        Math::T_Vector Network::getRolledGradient() {
            Math::T_RawVector tmp;

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

            Math::T_Vector result = Math::rawToVector(tmp);
            return result;
        }

        void Network::setRolledTheta(Math::T_Vector theta) {
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

        double Network::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
            return this->layers.at(this->getSize() - 1)->loss(std::move(output), std::move(predictions));
        }

        double Network::error(T_Size m) {
            return this->layers.at(this->getSize() - 1)->error(m);
        }

        void Network::debug() {
            this->layers.at(0)->debug();
        }
    }
}
