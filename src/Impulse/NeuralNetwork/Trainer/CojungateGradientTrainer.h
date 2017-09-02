#ifndef COJUNGATE_GTADIENT_TRAINER
#define COJUNGATE_GTADIENT_TRAINER

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

#endif //COJUNGATE_GTADIENT_TRAINER
