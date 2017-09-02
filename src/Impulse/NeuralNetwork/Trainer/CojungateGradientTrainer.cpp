#include "CojungateGradientTrainer.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::NeuralNetwork::Math::T_Vector;
using Impulse::NeuralNetwork::Network;
using Impulse::NeuralNetwork::Math::Fmincg;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            void ConjugateGradientTrainer::train(Impulse::SlicedDataset &dataSet) {
                Fmincg minimizer;
                Network *network = this->network;
                T_Vector theta = network->getRolledTheta();
                double regularization = this->regularization;

                network->backward(dataSet.getInput(), dataSet.getOutput(), network->forward(dataSet.getInput()),
                                  this->regularization);

                StepFunction callback(
                        [this, &dataSet, &regularization](T_Vector input) {
                            this->network->setRolledTheta(input);
                            this->network->backward(dataSet.getInput(), dataSet.getOutput(),
                                                    this->network->forward(dataSet.getInput()), regularization);
                            return this->cost(dataSet);
                        });

                this->network->setRolledTheta(
                        minimizer.minimize(callback, theta, this->learningIterations, this->verbose));
            }
        }
    }
}
