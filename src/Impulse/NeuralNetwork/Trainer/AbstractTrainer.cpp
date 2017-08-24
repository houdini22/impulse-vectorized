#include "AbstractTrainer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            AbstractTrainer::AbstractTrainer(Impulse::NeuralNetwork::Network *net) {
                this->network = net;
            }

            Impulse::NeuralNetwork::Network *AbstractTrainer::getNetwork() {
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

            double AbstractTrainer::cost(Impulse::SlicedDataset &dataSet) {
                unsigned int m = dataSet.output.getSize();
                Eigen::MatrixXd A = this->network->forward(dataSet.getInput());
                Eigen::MatrixXd Y = dataSet.getOutput();

                Eigen::MatrixXd errors = (Y.array() * A.unaryExpr([](const double x) { return log(x); }).array())
                                         +
                                         (Y.unaryExpr([](const double x) { return 1.0 - x; }).array()
                                          *
                                          A.unaryExpr([](const double x) { return log(1.0 - x); }).array()
                                         );

                return (-1.0) / (double) m * errors.sum();
            }

            void AbstractTrainer::train(Impulse::SlicedDataset &dataSet) {
                Eigen::MatrixXd X = dataSet.getInput();
                Eigen::MatrixXd Y = dataSet.getOutput();
                Impulse::NeuralNetwork::Network *network = this->getNetwork();

                if (this->verbose) {
                    std::cout << "Starting training with " << this->learningIterations << " iterations." << std::endl;
                }

                for (unsigned int step = 0; step < this->learningIterations; step++) {
                    Eigen::MatrixXd predictions = network->forward(X);

                    network->backward(X, Y, predictions);

                    network->updateParameters(this->learningRate);

                    double cost = this->cost(dataSet);

                    if (true || this->verbose && step + 1 % this->verboseStep == 0) {
                        std::cout << "Iteration: " << step << " | Error:" << cost << std::endl;
                    }
                }

                if (this->verbose) {
                    std::cout << "Learning ended after " << this->learningIterations << " iterations "
                              << "with error = " << this->cost(dataSet) << "." << std::endl;
                }
            }
        }

    }

}