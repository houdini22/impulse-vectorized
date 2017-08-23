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
            public:
                Eigen::MatrixXd W;
                Eigen::VectorXd b;
                Eigen::MatrixXd dZ;

                Abstract(unsigned int size, unsigned int prevSize) {
                    this->size = size;
                    this->prevSize = prevSize;

                    this->W.resize(this->size, this->prevSize);
                    this->W.setRandom();

                    this->b.resize(this->size);
                    this->b.setZero();
                }

                virtual Eigen::MatrixXd forward(Eigen::MatrixXd input) = 0;

                virtual Eigen::MatrixXd backward(Eigen::MatrixXd A) = 0;

                virtual void updateParameters(double learningRate) = 0;

                virtual Eigen::MatrixXd derivative() = 0;

                virtual Eigen::MatrixXd activation(Eigen::MatrixXd &input) {
                    return Eigen::MatrixXd(input);
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
