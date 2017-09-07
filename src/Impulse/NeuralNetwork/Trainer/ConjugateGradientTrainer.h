#ifndef COJUNGATE_GTADIENT_TRAINER
#define COJUNGATE_GTADIENT_TRAINER

#include "../include.h"

using namespace Impulse::NeuralNetwork;

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
