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
                }

                Eigen::MatrixXd forward(Eigen::MatrixXd input) {
                    Eigen::MatrixXd Z = this->W * input;
                    Z += this->b;
                    this->A = this->activation(Z);
                    return this->A;
                }

                Eigen::MatrixXd activation(Eigen::MatrixXd &input) {
                    Eigen::MatrixXd result(input);
                    for (unsigned int i = 0; i < input.size(); i++) {
                        result(i) = 1.0 / (1.0 + exp(-input(i)));
                    }
                    return result;
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_SIGMOID_H
