#ifndef IMPULSE_VECTORIZED_ABSTRACT_H
#define IMPULSE_VECTORIZED_ABSTRACT_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract {
            protected:
                unsigned int size; // number of neurons
                unsigned int prevSize = 0; // number of prev layer size (input)
            public:
                Eigen::MatrixXd W; // weights
                Eigen::VectorXd b; // bias
                Eigen::MatrixXd A; // output of the layer after activation
                Eigen::MatrixXd Z; // output of the layer before activation
                Eigen::MatrixXd dW; // deltas for weights
                Eigen::MatrixXd db; // deltas for bias
                Eigen::MatrixXd gW; // gradient for weights
                Eigen::MatrixXd gb; // gradient for bias (filled by ones)

                Abstract(unsigned int size, unsigned int prevSize) {
                    this->size = size;
                    this->prevSize = prevSize;

                    // initialize weights
                    this->W.resize(this->size, this->prevSize);
                    this->W.setRandom();
                    this->W = this->W.array() * sqrt(2.0 / this->prevSize);

                    // initialize bias
                    this->b.resize(this->size, 1);
                    this->b.setZero();

                    // initialize gradient for weights
                    this->gW.resize(this->size, this->prevSize);

                    // initialize gradient for bias
                    this->gb.resize(this->size, 1);
                    this->gb.setOnes(); // always 1 since its bias
                }

                /**
                 * Forward propagation.
                 * @param input
                 * @return
                 */
                Eigen::MatrixXd forward(Eigen::MatrixXd input) {
                    this->Z = (this->W * input).colwise() + this->b;
                    this->A = this->activation(this->Z);
                    return this->A;
                }

                /**
                 * Calculates activated values.
                 * @param input
                 * @return
                 */
                virtual Eigen::MatrixXd activation(Eigen::MatrixXd input) = 0;

                /**
                 * Calculates derivative. It depends on activation function.
                 * @return
                 */
                virtual Eigen::MatrixXd derivative() = 0;

                void updateParameters(double learningRate) {
                    this->W -= learningRate * this->dW;
                    this->b -= learningRate * this->db;
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
                 * @param backwardInput
                 * @return
                 */
                Eigen::MatrixXd calculateDerivative(Eigen::MatrixXd backwardInput) {
                    this->gW = backwardInput.array() * this->derivative().array();
                    return this->gW;
                }

                /**
                 * Calculates deltas.
                 * @param backwardInput
                 * @param prevA
                 * @param regularization
                 * @param m
                 */
                void calculateDeltas(Eigen::MatrixXd backwardInput, Eigen::MatrixXd prevA, double regularization, double m) {
                    this->dW = backwardInput * prevA.transpose() + (regularization / m * this->W);
                    this->db = backwardInput.rowwise().sum();
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
