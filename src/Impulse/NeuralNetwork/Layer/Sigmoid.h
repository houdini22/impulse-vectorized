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

                Eigen::MatrixXd forward(Eigen::MatrixXd input) {
#ifdef DEBUG
                    std::cout << "input:" << std::endl << input << std::endl << std::endl;
                    std::cout << "W: " << std::endl << this->W
                              << std::endl; //this->W.rows() << "," << this->W.cols() << std::endl;
                    std::cout << "b: " << std::endl << this->b
                              << std::endl; //this->b.rows() << "," << this->b.cols() << std::endl;
#endif
                    this->Z = (this->W * input).colwise() + this->b;
                    this->A = this->activation(this->Z);
                    //this->dZ = this->A.array() * this->derivative().array();
#ifdef DEBUG
                    std::cout << "Z: " << this->Z << std::endl;
                    std::cout << "A: " << this->A << std::endl;
                    std::cout << "dZ: " << this->dZ << std::endl;
                    std::cout << "---" << std::endl << std::endl;
#endif
                    return this->A;
                }

                Eigen::MatrixXd activation(Eigen::MatrixXd input) {
                    Eigen::MatrixXd result = input.unaryExpr([](const double x) { return 1.0 / (1.0 + exp(-x)); });
                    return result;
                }

                Eigen::MatrixXd backward(Eigen::MatrixXd A) {
                    // num examples
                    long m = A.cols();

                    std::cout << "A:" << std::endl << A << std::endl << std::endl;
                    std::cout << "this->A:" << std::endl << this->A << std::endl << std::endl;
                    std::cout << "this->W:" << std::endl << this->W << std::endl << std::endl;

                    Eigen::MatrixXd dZ = A.array() * this->derivative(A).array();

                    this->dW = (1.0 / (double) m) * (dZ * this->A.transpose());
                    this->db = (1.0 / (double) m) * (dZ.rowwise().sum());

                    Eigen::MatrixXd result = (this->W.transpose() * dZ);

                    std::cout << "this->dW:" << std::endl << this->dW << std::endl << std::endl;
                    std::cout << "dZ:" << std::endl << dZ << std::endl << std::endl;
                    //std::cout << "b:" << std::endl << this->b << std::endl << std::endl;
                    //std::cout << "db:" << std::endl << this->db << std::endl << std::endl;

                    return result;
                }

                Eigen::MatrixXd derivative(Eigen::MatrixXd A) {
                    return A.array() * (1.0 - A.array());
                }

                void updateParameters(double learningRate) {
                    //std::cout << "this->W:" << std::endl << this->W << std::endl << std::endl;
                    //std::cout << "this->dW:" << std::endl << this->dW << std::endl << std::endl;
                    //this->W = this->W.array() - (learningRate * this->dW.array());
                    //this->b = this->b.array() - (learningRate * this->db.array());
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_SIGMOID_H
