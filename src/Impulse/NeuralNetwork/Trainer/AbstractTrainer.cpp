#include "AbstractTrainer.h"

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
                T_Matrix A = this->network->forward(dataSet.getInput());
                T_Matrix Y = dataSet.getOutput();

                /*T_Matrix errors = (Y.array() * A.unaryExpr([](const double x) { return log(x); }).array())
                                  +
                                  (Y.unaryExpr([](const double x) { return 1.0 - x; }).array()
                                   *
                                   A.unaryExpr([](const double x) { return log(1.0 - x); }).array()
                                  );*/

                T_Matrix errors = (Y.array() * A.unaryExpr([](const double x) { return log(x); }).array());

                double p = 0.0;
                for (T_Size i = 0; i < this->network->getSize(); i++) {
                    p += this->network->getLayer(i)->W.unaryExpr([](const double x) {
                        return pow(x, 2.0);
                    }).sum();
                }

                double error = (-1.0 / (double) m) * errors.sum() + ((this->regularization * p) / (2.0 * (double) m));

                Impulse::NeuralNetwork::Trainer::CostGradientResult result;
                result.error = error;
                result.gradient = this->network->getRolledGradient();

                return result;
            }
        }
    }
}