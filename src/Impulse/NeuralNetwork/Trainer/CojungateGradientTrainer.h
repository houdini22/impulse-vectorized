#ifndef IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H
#define IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H

#include "AbstractTrainer.h"
#include "../Math/Minimizer/Fmincg.h"
#include "../Network.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            typedef std::function<Impulse::NeuralNetwork::Trainer::CostGradientResult(
                    Eigen::VectorXd)> CostFunction;

            class ConjugateGradientTrainer : public AbstractTrainer {
            public:
                ConjugateGradientTrainer(Network *net) : AbstractTrainer(net) {

                }

                void train(Impulse::SlicedDataset &dataSet) {
                    Impulse::NeuralNetwork::Math::Minimizer::Fmincg minimizer;
                    Impulse::NeuralNetwork::Network *network = this->network;
                    Eigen::VectorXd theta = network->getRolledTheta();
                    double regularization = this->regularization;

                    network->backward(dataSet.getInput(), dataSet.getOutput(), network->forward(dataSet.getInput()), this->regularization);

                    CostFunction callback(
                            [this, &dataSet, &regularization](Eigen::VectorXd input) {
                                //this->network->setRolledTheta(input);
                                //this->network->backward(dataSet.getInput(), dataSet.getOutput(), this->network->forward(dataSet.getInput()), regularization);
                                return this->cost(dataSet);
                            });

                    this->network->setRolledTheta(minimizer.minimize(callback, theta, this->learningIterations, true));
                }
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H
