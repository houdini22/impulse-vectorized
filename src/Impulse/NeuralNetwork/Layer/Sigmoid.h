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
                Eigen::MatrixXd A;
                Eigen::MatrixXd Z;
                Eigen::MatrixXd dW;
                Eigen::MatrixXd db;
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

                    this->dW = (1.0 / (double) m) * (A * this->A.transpose());
                    this->db = (1.0 / (double) m) * (A.colwise().sum());

                    Eigen::MatrixXd result = (this->W.transpose() * A).unaryExpr(
                            [](const double x) { return 1.0 - pow(x, 2.0); });

                    std::cout << "A: " << A.rows() << "," << A.cols() << std::endl;
                    std::cout << "this->A: " << this->A.rows() << "," << this->A.cols() << std::endl;
                    std::cout << "W: " << this->W.rows() << "," << this->W.cols() << std::endl;
                    std::cout << "dW: " << this->dW.rows() << "," << this->dW.cols() << std::endl;
                    std::cout << "b: " << this->db.rows() << "," << this->db.cols() << std::endl;
                    std::cout << "db: " << this->db.rows() << "," << this->db.cols() << std::endl;

                    return result;
                }

                Eigen::MatrixXd derivative() {
                    return this->Z.array() * (1.0 - this->Z.array());
                }

                void updateParameters(double learningRate) {
                    this->W = this->W - (learningRate * this->dW);
                    std::cout << "OLD B:" <<
                              std::endl << this->b << std::endl << "OLD DB:" << std::endl << this->db << std::endl;
                    this->b = this->b - (learningRate * this->db);
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_SIGMOID_H
