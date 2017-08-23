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
                              << std::endl;
                    std::cout << "b: " << std::endl << this->b
                              << std::endl;

                    std::cout << "TEST:" << std::endl << (this->W * input).colwise() + this->b << std::endl << std::endl;
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

                Eigen::MatrixXd backward(Eigen::MatrixXd dA, Eigen::MatrixXd prevA) {
                    // num examples
                    long m = dA.cols();

                    Eigen::MatrixXd dZ = dA.array() * this->derivative(this->Z).array();

                    this->dW = (1.0 / (double) m) * (dZ * prevA.transpose());
                    this->db = (1.0 / (double) m) * (dZ.rowwise().sum());

                    Eigen::MatrixXd prevDA = (this->W.transpose() * dZ);

                    assert(this->size == this->A.rows());
                    assert(m == this->A.cols());

                    assert(this->size == this->Z.rows());
                    assert(m == this->Z.cols());

                    assert(this->size == dZ.rows());
                    assert(m == dZ.cols());

                    assert(this->size == this->W.rows());
                    assert(this->prevSize == this->W.cols());

                    assert(this->size == this->b.rows());
                    assert(this->b.cols() == 1);

                    assert(this->W.rows() == this->dW.rows());
                    assert(this->W.cols() == this->dW.cols());

                    assert(this->b.rows() == this->db.rows());
                    assert(this->b.cols() == this->db.cols());
                    //std::cout << "m:" << std::endl << m << std::endl << std::endl;
                    //std::cout << "this->A:" << std::endl << this->A << std::endl << std::endl;
                    //std::cout << "this->Z:" << std::endl << this->Z << std::endl << std::endl;
                    //std::cout << "A:" << std::endl << A << std::endl << std::endl;
                    //std::cout << "this->W:" << std::endl << this->W << std::endl << std::endl;
                    //std::cout << "b:" << std::endl << this->b << std::endl << std::endl;
                    //std::cout << "dZ:" << std::endl << dZ << std::endl << std::endl;
                    //std::cout << "result:" << std::endl << result << std::endl << std::endl;
                    //std::cout << "this->dW:" << std::endl << this->dW << std::endl << std::endl;
                    //std::cout << "---" << std::endl << std::endl << std::endl;
                    //std::cout << "A: " << std::endl << A << std::endl << std::endl;
                    //std::cout << "dZ:" << std::endl << dZ << std::endl << std::endl;
                    //std::cout << "result:" << std::endl << result << std::endl << std::endl;
                    //std::cout << "db:" << std::endl << this->db << std::endl << std::endl;

                    return prevDA;
                }

                Eigen::MatrixXd derivative(Eigen::MatrixXd Z) {
                    return Z.array() * (1.0 - Z.array());
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
