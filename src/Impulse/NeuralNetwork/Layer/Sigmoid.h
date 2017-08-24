#ifndef IMPULSE_VECTORIZED_SIGMOID_H
#define IMPULSE_VECTORIZED_SIGMOID_H

#include <string>
#include "Abstract.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const std::string TYPE_SIGMOID = "sigmoid";

            class Sigmoid : public Abstract {
            protected:
            public:

                Sigmoid(unsigned int size, unsigned int prevSize) : Abstract(size, prevSize) {

                }

                Eigen::MatrixXd activation(Eigen::MatrixXd input) {
                    Eigen::MatrixXd result = input.unaryExpr([](const double x) { return 1.0 / (1.0 + exp(-x)); });
                    return result;
                }

                Eigen::MatrixXd backward(Eigen::MatrixXd dZ, Eigen::MatrixXd prevA) {
                    // num examples
                    long m = dZ.cols();

                    this->dW = (1.0 / (double) m) * (dZ * prevA.transpose());
                    this->db = (1.0 / (double) m) * (dZ.rowwise().sum());

                    Eigen::MatrixXd result = this->W.transpose() * dZ;

                    assert(this->size == this->A.rows());
                    assert(m == this->A.cols());

                    assert(this->size == this->Z.rows());
                    assert(m == this->Z.cols());

                    assert(this->size == this->W.rows());
                    assert(this->prevSize == this->W.cols());

                    assert(this->size == this->b.rows());
                    assert(this->b.cols() == 1);

                    assert(this->W.rows() == this->dW.rows());
                    assert(this->W.cols() == this->dW.cols());

                    assert(this->b.rows() == this->db.rows());
                    assert(this->b.cols() == this->db.cols());

                    return result;
                }

                Eigen::MatrixXd derivative() {
                    return this->A.array() * (1.0 - this->A.array());
                }

                void updateParameters(double learningRate) {
                    this->W = this->W.array() - (learningRate * this->dW.array());
                    this->b = this->b.array() - (learningRate * this->db.array());
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_SIGMOID_H
