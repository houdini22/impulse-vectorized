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
                Eigen::VectorXd b;
                Eigen::MatrixXd A;
                Eigen::MatrixXd Z;
                Eigen::MatrixXd dW;
                Eigen::MatrixXd db;
            public:
                Eigen::MatrixXd dA;

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
                    this->Z.resize(this->W.rows(), this->A.cols());

                    Eigen::MatrixXd Z = this->W * input;

                    this->Z = Z.colwise() + this->b;
                    this->A = this->activation(Z);
                    this->dZ = this->A.array() * this->derivative().array();

                    assert(this->Z.rows() == this->W.rows());
                    assert(this->Z.cols() == this->A.cols());
                    assert(this->A.rows() == this->W.rows());
                    assert(this->A.cols() == input.cols());

                    return this->A;
                }

                Eigen::MatrixXd activation(Eigen::MatrixXd &input) {
                    Eigen::MatrixXd result = input.unaryExpr([](const double x) { return 1.0 / (1.0 + exp(-x)); });
                    return result;
                }

                void backward(Impulse::NeuralNetwork::Layer::Abstract *nextLayer) {
                    this->dW.resize(0, 0);
                    this->db.resize(0, 0);
                    //this->dZ.resize(0, 0);

                    //std::cout << this->dZ.rows() << "," << this->dZ.cols() << std::endl;
                    //std::cout << nextLayer->dA.rows() << "," << nextLayer->dA.cols() << std::endl;

                    this->dW = (1.0 / (double) this->dA.cols()) * (this->dZ * nextLayer->dA.transpose());
                    this->db = (1.0 / (double) this->dA.cols()) * (this->dZ.colwise().sum());
                    this->dA = nextLayer->W.transpose() * nextLayer->dZ;
                }

                Eigen::MatrixXd derivative() {
                    return this->Z.array() * (1.0 - this->Z.array());
                }

                void updateParameters(double learningRate) {
                    std::cout << this->W.rows() << "," << this->W.cols() << std::endl;
                    std::cout << this->dW.rows() << "," << this->dW.cols() << std::endl;
                    this->W = this->W - (learningRate * this->dW);
                    this->b = this->b - (learningRate * this->db);
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_SIGMOID_H
