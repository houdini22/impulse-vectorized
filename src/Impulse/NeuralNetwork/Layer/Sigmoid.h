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
                    this->W.setZero();

                    this->b.resize(this->size);
                    this->b.setZero();
                }

                Eigen::MatrixXd forward(Eigen::MatrixXd input) {
#ifdef DEBUG
                    std::cout << "input:" << std::endl << input << std::endl << std::endl;
                    std::cout << "W: " << std::endl << this->W
                              << std::endl; //this->W.rows() << "," << this->W.cols() << std::endl;
                    std::cout << "b: " << std::endl << this->b
                              << std::endl; //this->b.rows() << "," << this->b.cols() << std::endl;
                    /*//std::cout << "Product: " << std::endl << (this->W.array().rowwise() * input.array()) << std::endl;
                    std::cout << "Product2: " << std::endl
                              << (this->W.transpose().array().colwise() * input.col(0).array()) << std::endl;
                    std::cout << "Product2: " << std::endl
                              << (this->W.transpose().array().colwise() * input.col(0).array()).colwise().sum()
                              << std::endl;
                    std::cout << "Product2: " << std::endl
                              << (this->W.transpose().array().colwise() * input.col(0).array()).colwise().sum().matrix() +
                                 this->b.transpose() << std::endl;
                    std::cout << "---" << std::endl;*/
#endif
                    this->Z = ((this->W.transpose().array().colwise() * input.col(0).array()).colwise().sum().matrix() +
                               this->b.transpose()).transpose();
                    this->A = this->activation(this->Z);
                    this->dZ = this->A.array() * this->derivative().array();
#ifdef DEBUG
                    std::cout << "Z: " << this->Z << std::endl;
                    std::cout << "A: " << this->A << std::endl;
                    std::cout << "dZ: " << this->dZ << std::endl;
                    std::cout << "---" << std::endl << std::endl;
#endif
                    return this->A;
                }

                Eigen::MatrixXd activation(Eigen::MatrixXd &input) {
                    Eigen::MatrixXd result = input.unaryExpr([](const double x) { return 1.0 / (1.0 + exp(-x)); });
                    return result;
                }

                void backward(Impulse::NeuralNetwork::Layer::Abstract *nextLayer) {
                    //this->dW.resize(0, 0);
                    //this->db.resize(0, 0);
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
