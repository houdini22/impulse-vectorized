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

            void AbstractTrainer::setRegularization(double regularization) {
                this->regularization = regularization;
            }

            void AbstractTrainer::setLearningIterations(unsigned int nb) {
                this->learningIterations = nb;
            }

            Impulse::NeuralNetwork::Trainer::CostGradientResult AbstractTrainer::cost(Impulse::SlicedDataset &dataSet) {
                unsigned int m = dataSet.output.getSize();
                Eigen::MatrixXd AL = this->network->forward(dataSet.getInput());
                Eigen::MatrixXd Y = dataSet.getOutput();

                double error = -(1.0 / (double) m) * (double) (
                        (Y * AL.transpose().unaryExpr([](const double x) { return log(x); }))
                        *
                        ((Y.unaryExpr([](const double x) { return 1.0 - x; })) *
                         AL.transpose().unaryExpr([](const double x) { return log(1.0 - x); }))
                ).sum();

                CostGradientResult result;
                result.error = error;

                return result;
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
                    network->backward(predictions, Y);

                    network->updateParameters(this->learningRate);

                    CostGradientResult result = this->cost(dataSet);

                    if (this->verbose) {
                        std::cout << "Iteration: " << step << " | Error:" << result.getCost() << std::endl;
                    }
                }

                if (this->verbose) {
                    std::cout << "Learning ended." << std::endl;
                }
            }
        }

    }

}