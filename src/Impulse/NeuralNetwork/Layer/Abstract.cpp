#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract::Abstract(T_Size size, T_Size prevSize) {
                this->size = size;
                this->prevSize = prevSize;

                // initialize weights
                this->W.resize(this->size, this->prevSize);
                this->W.setRandom();
                this->W = this->W * sqrt(2.0 / this->prevSize);

                // initialize bias
                this->b.resize(this->size);
                this->b.setZero();
            }

            Math::T_Matrix Abstract::forward(Math::T_Matrix input) {
                this->Z = (this->W * input).colwise() + this->b;
                return this->A = this->activation();
            }

            T_Size Abstract::getSize() {
                return this->size;
            }

            T_Size Abstract::getOutputSize() {
                return this->size;
            }
        }
    }
}
