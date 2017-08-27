#ifndef IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H
#define IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H

#include "AbstractTrainer.h"
#include "../Math/Minimizer/Fmincg.h"
#include "../Network.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            class ConjugateGradientTrainer : public AbstractTrainer {
            public:
                ConjugateGradientTrainer(Network *net) : AbstractTrainer(net) {

                }

                void train(Impulse::SlicedDataset &dataSet) {
                    Impulse::NeuralNetwork::Math::Minimizer::Fmincg minimizer;

                    Impulse::NeuralNetwork::Network *network = this->network;
                    Eigen::VectorXd theta = network->getRolledTheta();

                    std::cout << theta.rows() << "," << theta.cols() << std::endl;

                    std::function<Impulse::NeuralNetwork::Trainer::CostGradientResult(Eigen::VectorXd)> callback(
                            [this, &dataSet](Eigen::VectorXd input) {
                                std::cout << input.rows() << "," << input.cols() << std::endl;
                                this->network->setRolledTheta(input);
                                return this->cost(dataSet);
                            });

                    minimizer.minimize(callback, theta, this->learningIterations, true);
                    //this->network->setRolledTheta(minimizer.minimize(cf, theta, this->learningIterations, true));
                }
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H
