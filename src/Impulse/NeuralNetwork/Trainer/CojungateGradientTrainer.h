#ifndef IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H
#define IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H

#include "AbstractTrainer.h"
#include "../Math/Minimizer/Fmincg.h"
#include "../Network.h"
#include "../Math/Matrix.h"

using Matrix = Impulse::NeuralNetwork::Math::T_Matrix;
using Vector = Impulse::NeuralNetwork::Math::T_Vector;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            typedef std::function<Impulse::NeuralNetwork::Trainer::CostGradientResult(
                    Vector)> CostFunction;

            class ConjugateGradientTrainer : public AbstractTrainer {
            public:
                ConjugateGradientTrainer(Network *net) : AbstractTrainer(net) {

                }

                void train(Impulse::SlicedDataset &dataSet) {
                    Impulse::NeuralNetwork::Math::Minimizer::Fmincg minimizer;
                    Impulse::NeuralNetwork::Network *network = this->network;
                    Vector theta = network->getRolledTheta();
                    double regularization = this->regularization;

                    network->backward(dataSet.getInput(), dataSet.getOutput(), network->forward(dataSet.getInput()), this->regularization);

                    CostFunction callback(
                            [this, &dataSet, &regularization](Vector input) {
                                this->network->setRolledTheta(input);
                                this->network->backward(dataSet.getInput(), dataSet.getOutput(), this->network->forward(dataSet.getInput()), regularization);
                                return this->cost(dataSet);
                            });

                    this->network->setRolledTheta(minimizer.minimize(callback, theta, this->learningIterations, this->verbose));
                }
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_CONJUGATEGRADIENTTRAINER_H
