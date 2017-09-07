#ifndef IMPULSE_NEURALNETWORK_TRAINER_ABSTRACT_H
#define IMPULSE_NEURALNETWORK_TRAINER_ABSTRACT_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class AbstractTrainer {
            protected:
                Network *network;
                double regularization = 0.0;
                T_Size learningIterations = 1000;
                double learningRate = 0.1;
                bool verbose = true;
                int verboseStep = 100;
            public:
                AbstractTrainer(Network *net);

                Network *getNetwork();

                void setRegularization(double value);

                void setLearningIterations(T_Size value);

                void setLearningRate(double value);

                void setVerbose(bool value);

                void setVerboseStep(int value);

                CostGradientResult cost(Impulse::SlicedDataset &dataSet);

                virtual void train(Impulse::SlicedDataset &dataSet) = 0;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_TRAINER_ABSTRACT_H
