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
                Eigen::VectorXd gradient;
                double &getError() {
                    return this->error;
                }
                Eigen::VectorXd &getGradient() {
                    return this->gradient;
                }
            };

            class AbstractTrainer {
            protected:
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

                Impulse::NeuralNetwork::Trainer::CostGradientResult cost(Impulse::SlicedDataset &dataSet);

                virtual void train(Impulse::SlicedDataset &dataSet) = 0;
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACTTRAINER_H
