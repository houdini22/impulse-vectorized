#ifndef IMPULSE_VECTORIZED_ABSTRACT_H
#define IMPULSE_VECTORIZED_ABSTRACT_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract {
            protected:
                unsigned int size;
                unsigned int prevSize = 0;
                Eigen::MatrixXd W;
                Eigen::VectorXd b;
                Eigen::MatrixXd A;
                Eigen::MatrixXd Z;
                Eigen::MatrixXd dW;
                Eigen::MatrixXd db;
            public:

                Abstract(unsigned int size, unsigned int prevSize) {
                    this->size = size;
                    this->prevSize = prevSize;

                    this->W.resize(this->size, this->prevSize);
                    this->W.setRandom();

                    this->b.resize(this->size);
                    this->b.setZero();
                }

                virtual Eigen::MatrixXd forward(Eigen::MatrixXd input) = 0;

                virtual Eigen::MatrixXd backward(Eigen::MatrixXd dA, Eigen::MatrixXd prevA) = 0;

                virtual void updateParameters(double learningRate) = 0;

                virtual Eigen::MatrixXd activation(Eigen::MatrixXd input) = 0;

                virtual Eigen::MatrixXd derivative(Eigen::MatrixXd A) = 0;

                Eigen::MatrixXd getA() {
                    return this->A;
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
