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

            void Abstract::configure() {
                // initialize weights
                this->W.resize(this->height, this->width);
                this->W.setRandom();
                this->W = this->W * sqrt(2.0 / this->width);

                // initialize bias
                this->b.resize(this->height);
                this->b.setZero();
            }

            void Abstract::transition(Layer::LayerPointer prevLayer) {
                // none by default
            }

            T_Size Abstract::getOutputHeight() {
                return this->height;
            }

            T_Size Abstract::getOutputWidth() {
                return this->width;
            }

            T_Size Abstract::getOutputDepth() {
                return 1;
            }
        }
    }
}
