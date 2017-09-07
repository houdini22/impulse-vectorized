#ifndef TRAINER_TYPES_H
#define TRAINER_TYPES_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            struct CostGradientResult {
                double error;
                Math::T_Vector gradient;

                double &getError() {
                    return this->error;
                }

                Math::T_Vector &getGradient() {
                    return this->gradient;
                }
            };

            typedef std::function<CostGradientResult(
                    Math::T_Vector)> StepFunction;
        }
    }
}

#endif //IMPULSE_VECTORIZED_TYPES_H
