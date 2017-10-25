#ifndef IMPULSE_NEURALNETWORK_TRAINER_CONJUGATE_GRADIENT_TRAINER_H
#define IMPULSE_NEURALNETWORK_TRAINER_CONJUGATE_GRADIENT_TRAINER_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class ConjugateGradientTrainer : public AbstractTrainer {
            public:
                explicit ConjugateGradientTrainer(Network::Abstract &net);

                void train(Impulse::SlicedDataset &dataSet) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_TRAINER_CONJUGATE_GRADIENT_TRAINER_H
