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
                Eigen::MatrixXd dA;
                Eigen::MatrixXd dZ;

                Abstract(unsigned int size) {
                    this->size = size;
                }

                virtual Eigen::MatrixXd forward(Eigen::MatrixXd input) = 0;

                virtual void backward(Impulse::NeuralNetwork::Layer::Abstract *prevLayer) = 0;

                virtual void updateParameters(double learningRate) = 0;

                virtual Eigen::MatrixXd derivative() = 0;

                Abstract *backward(Eigen::MatrixXd predictions, Eigen::MatrixXd Y) {
                    this->dA =
                            (Y.array() / predictions.array()) +
                            ((Y.unaryExpr([](const double x) { return 1.0 - x; }).array()))
                            /
                            (
                                    predictions.unaryExpr([](const double x) { return 1.0 - x; }).array()
                            );

                    return this;
                }

                virtual Eigen::MatrixXd activation(Eigen::MatrixXd &input) {
                    return Eigen::MatrixXd(input);
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
