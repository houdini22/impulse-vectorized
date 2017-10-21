#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            ConjugateGradientTrainer::ConjugateGradientTrainer(Network &net) : AbstractTrainer(net) {}

            void ConjugateGradientTrainer::train(Impulse::SlicedDataset &dataSet) {
                Math::Fmincg minimizer;
                Network network = this->network;
                Math::T_Vector theta = network.getRolledTheta();
                double regularization = this->regularization;

                network.backward(dataSet.getInput(), dataSet.getOutput(), network.forward(dataSet.getInput()),
                                 this->regularization);

                Trainer::StepFunction callback(
                        [this, &dataSet, &regularization](Math::T_Vector input) {
                            this->network.setRolledTheta(input);
                            this->network.backward(dataSet.getInput(), dataSet.getOutput(),
                                                   this->network.forward(dataSet.getInput()), regularization);
                            return this->cost(dataSet);
                        });

                this->network.setRolledTheta(
                        minimizer.minimize(callback, theta, this->learningIterations, this->verbose));
            }
        }
    }
}
