#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Network {

            Abstract::Abstract(T_Dimension dim) {
                this->dimension = dim;
            }

            void Abstract::addLayer(Layer::LayerPointer layer) {
                this->size++;
                this->layers.push_back(layer);
            }

            Math::T_Matrix Abstract::forward(const Math::T_Matrix &input) {
                Math::T_Matrix output = input;

                for (auto &layer : this->layers) {
                    output = layer->forward(output);
                }

                return output;
            }

            void Abstract::backward(Math::T_Matrix X, Math::T_Matrix Y, Math::T_Matrix predictions, double regularization) {
                T_Size m = Math::Matrix::cols(X);
                Math::T_Matrix delta = Math::Matrix::subtract(predictions, Y);

                for (auto it = this->layers.rbegin(); it != this->layers.rend(); ++it) {
                    auto layer = (*it);
                    delta = layer->backpropagation->propagate(X, m, regularization, delta);
                }
            }

            T_Dimension Abstract::getDimension() {
                return this->dimension;
            }

            T_Size Abstract::getSize() {
                return this->size;
            }

            Layer::LayerPointer Abstract::getLayer(T_Size key) {
                return this->layers.at(key);
            }

            Math::T_ColVector Abstract::getRolledTheta() {
                Math::T_RawVector tmp;

                for (T_Size i = 0; i < this->getSize(); i++) {
                    auto layer = this->getLayer(i);

                    if (layer->getType() == Layer::TYPE_MAXPOOL) {
                        continue;
                    }

                    tmp.reserve((unsigned long) (layer->W.n_cols * layer->W.n_rows) + (layer->b.n_cols * layer->b.n_rows));

                    for (T_Size j = 0; j < layer->W.n_rows; j++) {
                        for (T_Size k = 0; k < layer->W.n_cols; k++) {
                            tmp.push_back(layer->W(j, k));
                        }
                    }

                    for (T_Size j = 0; j < layer->b.n_rows; j++) {
                        for (T_Size k = 0; k < layer->b.n_cols; k++) {
                            tmp.push_back(layer->b(j, k));
                        }
                    }
                }

                Math::T_ColVector result = Math::rawToVector(tmp);
                return result;
            }

            Math::T_ColVector Abstract::getRolledGradient() {
                Math::T_RawVector tmp;

                for (T_Size i = 0; i < this->getSize(); i++) {
                    auto layer = this->layers.at(i);

                    if (layer->getType() == Layer::TYPE_MAXPOOL) {
                        continue;
                    }

                    for (T_Size j = 0; j < layer->gW.n_rows; j++) {
                        for (T_Size k = 0; k < layer->gW.n_cols; k++) {
                            tmp.push_back(layer->gW(j, k));
                        }
                    }

                    for (T_Size j = 0; j < layer->gb.n_rows; j++) {
                        for (T_Size k = 0; k < layer->gb.n_cols; k++) {
                            tmp.push_back(layer->gb(j, k));
                        }
                    }
                }

                Math::T_ColVector result = Math::rawToVector(tmp);
                return result;
            }

            void Abstract::setRolledTheta(Math::T_ColVector theta) {
                unsigned long t = 0;

                for (T_Size i = 0; i < this->getSize(); i++) {
                    auto layer = this->layers.at(i);

                    if (layer->getType() == Layer::TYPE_MAXPOOL) {
                        continue;
                    }

                    for (T_Size j = 0; j < layer->W.n_rows; j++) {
                        for (T_Size k = 0; k < layer->W.n_cols; k++) {
                            layer->W(j, k) = theta(t++);
                        }
                    }

                    for (T_Size j = 0; j < layer->b.n_rows; j++) {
                        for (T_Size k = 0; k < layer->b.n_cols; k++) {
                            layer->b(j, k) = theta(t++);
                        }
                    }
                }
            }

            double Abstract::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                return this->layers.at(this->getSize() - 1)->loss(std::move(output), std::move(predictions));
            }

            double Abstract::error(T_Size m) {
                return this->layers.at(this->getSize() - 1)->error(m);
            }

            void Abstract::debug() {

            }
        }
    }
}
