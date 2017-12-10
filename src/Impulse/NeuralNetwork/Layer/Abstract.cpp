#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract::Abstract() = default;

            Math::T_Matrix Abstract::forward(const Math::T_Matrix &input) {
                this->Z = (this->W * input).colwise() + this->b;
                return this->A = this->activation();
            }

            void Abstract::setSize(T_Size value) {
                this->setHeight(value);
            }

            void Abstract::setSize(T_Size width, T_Size height, T_Size depth) {
                this->setWidth(width);
                this->setHeight(height);
                this->setDepth(depth);
            }

            void Abstract::setPrevSize(T_Size value) {
                this->setWidth(value);
            }

            void Abstract::setWidth(T_Size value) {
                this->width = value;
            }

            void Abstract::setHeight(T_Size value) {
                this->height = value;
            }

            void Abstract::setDepth(T_Size value) {
                this->depth = value;
            }

            T_Size Abstract::getSize() {
                return this->height;
            }

            T_Size Abstract::getOutputWidth() {
                return this->width;
            }

            T_Size Abstract::getOutputHeight() {
                return this->height;
            }

            T_Size Abstract::getOutputDepth() {
                return 1;
            }

            Math::T_Matrix Abstract::backward(
                    Math::T_Matrix &sigma,
                    const Layer::LayerPointer &prevLayer,
                    Math::T_Matrix prevActivations,
                    long &m,
                    double &regularization
            ) {

                Math::T_Matrix delta = sigma * prevActivations.transpose().conjugate();

                this->gW = delta.array() / m + (regularization / m * this->W.array());
                this->gb = sigma.rowwise().sum() / m;

                if (prevLayer != nullptr) {
                    Math::T_Matrix tmp1 = this->W.transpose() * sigma;
                    Math::T_Matrix tmp2 = prevLayer->derivative();

                    return tmp1.array() * tmp2.array();
                }
                return Math::T_Matrix();
            }
        }
    }
}
