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

            Impulse::NeuralNetwork::Trainer::CostGradientResult AbstractTrainer::cost(Impulse::Dataset::SlicedDataset &dataSet, bool rollGradient) {
                T_Size batchSize = 100;
                T_Size numberOfExamples = dataSet.getNumberOfExamples();
                auto numBatches = (T_Size) ceil((double) numberOfExamples / (double) batchSize);

                double cost = 0.0;
                double accuracy = 0.0;

                // calculate penalty
                double penalty = 0.0;
                for (T_Size i = 0; i < this->network.getSize(); i++) {
                    penalty += Math::Matrix::sum(Math::Matrix::pow(this->network.getLayer(i)->W, 2.0));
                }

                // calculate cost from mini-batches
                for (T_Size batch = 0, offset = 0; batch < numberOfExamples; batch += batchSize, offset++) {
                    Math::T_Matrix predictedOutput = this->network.forward(dataSet.getInput(offset, batchSize));
                    Math::T_Matrix correctOutput = dataSet.getOutput(offset, batchSize);

                    auto miniBatchSize = (T_Size) correctOutput.n_cols;

                    double loss = this->network.loss(correctOutput, predictedOutput); // loss for the mini-batch
                    double error = this->network.error(miniBatchSize); // error for the mini-batch

                    cost += (error * loss + ((this->regularization * penalty) / (2.0 * (double) miniBatchSize)))
                            /
                            // TODO: fix it
                            ((double) numBatches * ((double) miniBatchSize / (double) batchSize));

                    for (T_Size i = 0; i < predictedOutput.n_cols; i++) {

                        arma::uword index1 = predictedOutput.col(i).index_max();
                        arma::uword index2 = correctOutput.col(i).index_max();

                        if (index1 == index2) {
                            accuracy++;
                        }
                    }
                }

                Impulse::NeuralNetwork::Trainer::CostGradientResult result;
                result.cost = cost;
                result.accuracy = accuracy / (double) numberOfExamples * 100.0;
                if (rollGradient) {
                    result.gradient = this->network.getRolledGradient();
                }

                return result;
            }
        }
    }
}