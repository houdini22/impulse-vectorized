#ifndef IMPULSE_VECTORIZED_ABSTRACT_H
#define IMPULSE_VECTORIZED_ABSTRACT_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../Math/Matrix.h"

using Matrix = Impulse::NeuralNetwork::Math::T_Matrix;
using Vector = Impulse::NeuralNetwork::Math::T_Vector;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract {
            protected:
                unsigned int size; // number of neurons
                unsigned int prevSize = 0; // number of prev layer size (input)
            public:
                Matrix W; // weights
                Vector b; // bias
                Matrix A; // output of the layer after activation
                Matrix Z; // output of the layer before activation
                Matrix gW;

                Abstract(unsigned int size, unsigned int prevSize) {
                    this->size = size;
                    this->prevSize = prevSize;

                    // initialize weights
                    this->W.resize(this->size, this->prevSize);
                    this->W.setRandom();
                    this->W = this->W.array() * sqrt(2.0 / this->prevSize);

                    // initialize bias
                    this->b.resize(this->size);
                    this->b.setZero();
                }

                /**
                 * Forward propagation.
                 * @param input
                 * @return
                 */
                Matrix forward(Matrix input) {
                    this->Z = (this->W * input).colwise() + this->b;
                    this->A = this->activation(this->Z);
                    return this->A;
                }

                /**
                 * Calculates activated values.
                 * @param input
                 * @return
                 */
                virtual Matrix activation(Matrix input) = 0;

                /**
                 * Calculates derivative. It depends on activation function.
                 * @return
                 */
                virtual Matrix derivative(Matrix input) = 0;

                void updateParameters(double learningRate) {
                    /*this->W -= learningRate * this->wGradient;
                    this->b -= learningRate * this->bGradient;*/
                }

                /**
                 * Getter for layer size.
                 * @return
                 */
                unsigned int getSize() {
                    return this->size;
                }

                /**
                 * Getter for layer type.
                 * @return
                 */
                virtual std::string getType() = 0;
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
