#ifndef IMPULSE_NEURALNETWORK_TRAINER_ABSTRACT_H
#define IMPULSE_NEURALNETWORK_TRAINER_ABSTRACT_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class AbstractTrainer {
            protected:
                Network network;                        // network to train
                double regularization = 0.0;            // regularization (lambda) parameters
                T_Size learningIterations = 1000;       // number of learning iterations
                double learningRate = 0.1;              // learning rate
                bool verbose = true;                    // if display messages
                int verboseStep = 100;                  // step for displaying messages
            public:
                /**
                 * Constructor.
                 * @param net
                 */
                AbstractTrainer(Network &net);

                /**
                 * Sets regularization (lamdba) parameters.
                 * @param value
                 */
                void setRegularization(double value);

                /**
                 * Sets learning iterations parameters.
                 * @param value
                 */
                void setLearningIterations(T_Size value);

                /**
                 * Sets learning rate parameters.
                 * @param value
                 */
                void setLearningRate(double value);

                /**
                 * Sets if display iteration messages.
                 * @param value
                 */
                void setVerbose(bool value);

                /**
                 * Sets verbose step.
                 * @param value
                 */
                void setVerboseStep(int value);

                /**
                 * Computes cost and gradient for given dataset.
                 * @param dataSet
                 * @return
                 */
                CostGradientResult cost(Impulse::SlicedDataset &dataSet);

                /**
                 * Trains network with given dataset.
                 * @param dataSet
                 */
                virtual void train(Impulse::SlicedDataset &dataSet) = 0;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_TRAINER_ABSTRACT_H
