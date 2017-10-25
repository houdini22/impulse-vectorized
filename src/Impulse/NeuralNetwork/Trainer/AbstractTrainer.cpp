#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            AbstractTrainer::AbstractTrainer(Network::Abstract &net) : network(net) {}

            void AbstractTrainer::setRegularization(double value) {
                this->regularization = value;
            }

            void AbstractTrainer::setLearningIterations(T_Size value) {
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
                T_Size m = dataSet.output.getSize();
                Math::T_Matrix A = this->network.forward(dataSet.getInput());
                Math::T_Matrix Y = dataSet.getOutput();

                double loss = this->network.loss(Y, A);
                double error = this->network.error(m);

                double p = 0.0;
                for (T_Size i = 0; i < this->network.getSize(); i++) {
                    p += this->network.getLayer(i)->W.unaryExpr([](const double x) {
                        return pow(x, 2.0);
                    }).sum();
                }

                double cost = error * loss + ((this->regularization * p) / (2.0 * (double) m));

                Impulse::NeuralNetwork::Trainer::CostGradientResult result;
                result.cost = cost;
                result.gradient = this->network.getRolledGradient();

                return result;
            }
        }
    }
}