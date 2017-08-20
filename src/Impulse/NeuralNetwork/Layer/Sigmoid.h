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
                Eigen::MatrixXd W;
                Eigen::VectorXd b;
                Eigen::MatrixXd A;
                Eigen::MatrixXd Z;
                Eigen::MatrixXd dW;
                Eigen::MatrixXd db;
            public:
                Sigmoid(unsigned int size, unsigned int prevSize) : Abstract(size) {
                    this->prevSize = prevSize;
                    this->initialize();
                }

                void initialize() {
                    this->W.resize(this->size, this->prevSize);
                    this->W.setRandom();

                    this->b.resize(this->size);
                    this->b.setZero();

                    assert(this->W.cols() == this->prevSize);
                    assert(this->W.rows() == this->size);
                    assert(this->b.cols() == 1);
                    assert(this->b.rows() == this->size);
                }

                Eigen::MatrixXd forward(Eigen::MatrixXd input) {
                    this->Z.resize(0, 0);

                    Eigen::MatrixXd Z = this->W * input;
                    Z.colwise() += this->b;
                    this->Z = Z;
                    this->A = this->activation(Z);
                    return this->A;
                }

                Eigen::MatrixXd activation(Eigen::MatrixXd &input) {
                    Eigen::MatrixXd result = input.unaryExpr([](const double x) { return 1.0 / (1.0 + exp(-x)); });
                    return result;
                }

                Eigen::MatrixXd backward(Eigen::MatrixXd dA) {
                    this->dW.resize(0, 0);
                    this->db.resize(0, 0);

                    Eigen::MatrixXd dZ = dA.array() * this->derivative().array();
                    this->dW = (1.0 / (double) dA.cols()) * (dZ.array() * dA.array());
                    this->db = (1.0 / (double) dA.cols()) * (dZ.colwise().sum());

                    assert(dA.cols() == this->A.cols());
                    assert(dA.rows() == this->A.rows());
                    assert(this->dW.cols() == this->W.cols());
                    assert(this->dW.rows() == this->W.rows());
                    assert(this->db.cols() == this->b.cols());
                    assert(this->db.rows() == this->b.rows());

                    Eigen::MatrixXd result = this->W.transpose() * dZ;
                    return result;
                }

                Eigen::MatrixXd derivative() {
                    return this->Z.array() * (1.0 - this->Z.array());
                }

                void updateParameters(double learningRate) {
                    this->W = this->W - (learningRate * this->dW);
                    this->b = this->b - (learningRate * this->db);
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_SIGMOID_H
