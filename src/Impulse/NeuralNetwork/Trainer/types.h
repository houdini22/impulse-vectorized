#ifndef IMPULSE_VECTORIZED_TRAINER_TYPES_H
#define IMPULSE_VECTORIZED_TRAINER_TYPES_H

#include "../Math/types.h"

using Impulse::NeuralNetwork::Math::T_Vector;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            struct CostGradientResult {
                double error;
                T_Vector gradient;
                double getError() {
                    return this->error;
                }
                T_Vector getGradient() {
                    return this->gradient;
                }
            };

            typedef std::function<CostGradientResult(
                    T_Vector)> StepFunction;
        }
    }
}

#endif //IMPULSE_VECTORIZED_TYPES_H
