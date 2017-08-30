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
                Matrix wGradient; // deltas for weights
                Matrix bGradient; // deltas for bias
                Matrix wDerivative; // derivative for weights
                Matrix bDerivative; // derivative for bias (filled by ones)
                Matrix accumulator; // accumulator for gradient computation

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

                    // initialize gradient for weights
                    this->wDerivative.resize(this->size, this->prevSize);

                    // initialize gradient for bias
                    this->bDerivative.resize(this->size, 1);
                    this->bDerivative.setOnes(); // always 1 since its bias
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
                    this->W -= learningRate * this->wGradient;
                    this->b -= learningRate * this->bGradient;
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

                /**
                 * Calculates gradient for weights. Not for bias vector since its... bias - gradient always is
                 * equal 1.
                 * @param backwardActivation
                 * @return
                 */
                Matrix calculateDerivative(Matrix backwardActivation) {
                    this->wDerivative = backwardActivation.array() * this->derivative(this->A).array();
                    return this->wDerivative;
                }

                /**
                 * Calculates deltas.
                 * @param gradientPrev
                 * @param aPrev
                 * @param regularization
                 * @param m
                 */
                void calculateGradient(Matrix gradientPrev, Matrix aPrev, double regularization,
                                       double m) {
                    this->wGradient = gradientPrev * aPrev.transpose() + (regularization / m * this->W);
                    this->bGradient = gradientPrev.rowwise().sum();
                }

                void calculateAccumulation() {
                    this->accumulator;
                }

                void resetBackward(unsigned int m) {

                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
