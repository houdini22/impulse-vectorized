#ifndef IMPULSE_VECTORIZED_ABSTRACTTRAINER_H
#define IMPULSE_VECTORIZED_ABSTRACTTRAINER_H

#include <math.h>
#include "../Network.h"
#include "../../../Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            struct CostGradientResult {
                double error;

                double &getCost() {
                    return this->error;
                }
            };

            class AbstractTrainer {
                Impulse::NeuralNetwork::Network *network;
                double regularization = 0.0;
                unsigned int learningIterations = 1000;
                double learningRate = 0.1;
                bool verbose = true;
                int verboseStep = 100;
            public:
                AbstractTrainer(Impulse::NeuralNetwork::Network *net);

                Impulse::NeuralNetwork::Network *getNetwork();

                void setRegularization(double value);

                void setLearningIterations(unsigned int value);

                void setLearningRate(double value);

                void setVerbose(bool value);

                void setVerboseStep(int value);

                double cost(Impulse::SlicedDataset &dataSet);

                void train(Impulse::SlicedDataset &dataSet);
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACTTRAINER_H
