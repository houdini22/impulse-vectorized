#ifndef IMPULSE_VECTORIZED_INPUT_H
#define IMPULSE_VECTORIZED_INPUT_H

#include <string>
#include "Abstract.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const std::string TYPE_INPUT = "input";

            class Input : public Abstract {
            public:
                Input(unsigned int size) : Abstract(size) {

                }

                Eigen::MatrixXd forward(Eigen::MatrixXd input) {
                    return input;
                }
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_INPUT_H
