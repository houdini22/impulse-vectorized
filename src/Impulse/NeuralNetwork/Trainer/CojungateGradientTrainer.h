#ifndef IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H
#define IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H

#include "AbstractTrainer.h"
#include "../Math/Fmincg.h"
#include "../Network.h"
#include "../Math/types.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::NeuralNetwork::Math::T_Vector;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class ConjugateGradientTrainer : public AbstractTrainer {
            public:
                ConjugateGradientTrainer(Network *net) : AbstractTrainer(net) {

                }

                void train(Impulse::SlicedDataset &dataSet);
            };
        }
    }
}

#endif //IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H
