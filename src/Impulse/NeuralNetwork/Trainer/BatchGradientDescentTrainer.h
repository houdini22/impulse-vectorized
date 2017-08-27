#ifndef IMPULSE_VECTORIZED_GRADIENTDESCENTTRAINER_H
#define IMPULSE_VECTORIZED_GRADIENTDESCENTTRAINER_H

#include "AbstractTrainer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class BatchGradientDescent : public AbstractTrainer {
            public:
                BatchGradientDescent(Network * net) : AbstractTrainer(net) {

                }

                void train(Impulse::SlicedDataset &dataSet) {
                    Eigen::MatrixXd X = dataSet.getInput();
                    Eigen::MatrixXd Y = dataSet.getOutput();
                    Impulse::NeuralNetwork::Network *network = this->getNetwork();

                    if (this->verbose) {
                        std::cout << "Starting training with " << this->learningIterations << " iterations." << std::endl;
                    }

                    for (unsigned int step = 0; step < this->learningIterations; step++) {
                        Eigen::MatrixXd predictions = network->forward(X);

                        network->backward(X, Y, predictions, this->regularization);

                        network->updateParameters(this->learningRate);

                        Impulse::NeuralNetwork::Trainer::CostGradientResult cost = this->cost(dataSet);

                        if (this->verbose && (step + 1) % this->verboseStep == 0) {
                            std::cout << "Iteration: " << (step + 1) << " | Error:" << cost.getError() << std::endl;
                        }
                    }

                    if (this->verbose) {
                        std::cout << "Training ended after " << this->learningIterations << " iterations "
                                  << "with error = " << this->cost(dataSet).getError() << "." << std::endl;
                    }
                }
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_GRADIENTDESCENTTRAINER_H
