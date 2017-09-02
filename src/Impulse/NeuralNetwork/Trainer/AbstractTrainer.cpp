#include "AbstractTrainer.h"
#include "../Math/Matrix.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::NeuralNetwork::Network;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            AbstractTrainer::AbstractTrainer(Network *net) {
                this->network = net;
            }

            Network *AbstractTrainer::getNetwork() {
                return this->network;
            }

            void AbstractTrainer::setRegularization(double value) {
                this->regularization = value;
            }

            void AbstractTrainer::setLearningIterations(unsigned int value) {
                this->learningIterations = value;
            }

            void AbstractTrainer::setLearningRate(double value) {
                this->learningRate = value;
            }

            void AbstractTrainer::setVerbose(bool value) {
                this->verbose = value;
            }

            void AbstractTrainer::setVerboseStep(int value) {
                this->verboseStep = value;
            }

            Impulse::NeuralNetwork::Trainer::CostGradientResult AbstractTrainer::cost(Impulse::SlicedDataset &dataSet) {
                unsigned int m = dataSet.output.getSize();
                T_Matrix A = this->network->forward(dataSet.getInput());
                T_Matrix Y = dataSet.getOutput();

                T_Matrix errors = (Y.array() * A.unaryExpr([](const double x) { return log(x); }).array())
                                         +
                                         (Y.unaryExpr([](const double x) { return 1.0 - x; }).array()
                                          *
                                          A.unaryExpr([](const double x) { return log(1.0 - x); }).array()
                                         );

                double regularization = 0.0;
                for (unsigned int i = 0; i < this->network->getSize(); i++) {
                    regularization += this->network->getLayer(i)->W.unaryExpr([](const double x) {
                        return pow(x, 2.0);
                    }).sum();
                }

                double error = (-1.0) / (double) m * errors.sum() + (this->regularization / 2 * regularization);

                Impulse::NeuralNetwork::Trainer::CostGradientResult result;
                result.error = error;
                result.gradient = this->network->getRolledGradient();

                return result;
            }
        }

    }

}