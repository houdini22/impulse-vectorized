#ifndef IMPULSE_VECTORIZED_ABSTRACT_H
#define IMPULSE_VECTORIZED_ABSTRACT_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../Math/Matrix.h"
#include "../../types.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::NeuralNetwork::Math::T_Vector;
using Impulse::T_Size;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract {
            protected:
                T_Size size;        // number of neurons
                T_Size prevSize;    // number of prev layer size (input)
            public:
                T_Matrix W;         // weights
                T_Vector b;         // bias
                T_Matrix A;         // output of the layer after activation
                T_Matrix Z;         // output of the layer before activation
                T_Matrix gW;        // gradient for weights
                T_Vector gb;        // gradient for biases

                Abstract(T_Size size, T_Size prevSize) {
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
                T_Matrix forward(T_Matrix input) {
                    this->Z = (this->W * input).colwise() + this->b;
                    this->A = this->activation(this->Z);
                    return this->A;
                }

                /**
                 * Calculates activated values.
                 * @param input
                 * @return
                 */
                virtual T_Matrix activation(T_Matrix input) = 0;

                /**
                 * Calculates derivative. It depends on activation function.
                 * @return
                 */
                virtual T_Matrix derivative() = 0;

                /**
                 * Getter for layer size.
                 * @return
                 */
                T_Size getSize() {
                    return this->size;
                }

                /**
                 * Getter for layer type.
                 * @return
                 */
                virtual const std::string getType() = 0;
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
