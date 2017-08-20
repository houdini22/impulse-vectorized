#ifndef IMPULSE_VECTORIZED_ABSTRACT_H
#define IMPULSE_VECTORIZED_ABSTRACT_H

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract {
            protected:
                unsigned int size;
                unsigned int prevSize = 0;
            public:
                Abstract(unsigned int size) {
                    this->size = size;
                }

                virtual Eigen::MatrixXd forward(Eigen::MatrixXd input) = 0;

                virtual Eigen::MatrixXd activation(Eigen::MatrixXd & input) {
                    return Eigen::MatrixXd(input);
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
