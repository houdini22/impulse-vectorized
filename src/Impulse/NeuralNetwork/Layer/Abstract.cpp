#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract::Abstract() = default;

            Abstract::Abstract(T_Size size, T_Size prevSize) {
                this->setSize(size);
                this->setPrevSize(prevSize);
            }

            Math::T_Matrix Abstract::forward(const Math::T_Matrix &input) {
                this->Z = (this->W * input).colwise() + this->b;
                return this->A = this->activation();
            }

            void Abstract::setSize(T_Size value) {
                this->size = value;
            }

            void Abstract::setPrevSize(T_Size value) {
                this->prevSize = value;
            }

            T_Size Abstract::getSize() {
                return this->size;
            }

            void Abstract::configure() {
                // initialize weights
                this->W.resize(this->size, this->prevSize);
                this->W.setRandom();
                this->W = this->W * sqrt(2.0 / this->prevSize);

                // initialize bias
                this->b.resize(this->size);
                this->b.setZero();
            }

            void Abstract::transition(Layer::LayerPointer prevLayer) {
                // none by default
            }

            T_Size Abstract::getOutputRows() {
                return this->size;
            }

            T_Size Abstract::getOutputCols() {
                return this->prevSize;
            }

            T_Size Abstract::getDepth() {
                return 1;
            }
        }
    }
}
