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

            Impulse::NeuralNetwork::Trainer::CostGradientResult AbstractTrainer::cost(Impulse::Dataset::SlicedDataset &dataSet) {
                T_Size batchSize = 100;
                T_Size numberOfExamples = dataSet.getNumberOfExamples();
                T_Size numBatches = (T_Size) ceil((double) numberOfExamples / (double) batchSize);

                double cost = 0.0;

                // calculate penalty
                double penalty = 0.0;
                for (T_Size i = 0; i < this->network.getSize(); i++) {
                    penalty += this->network.getLayer(i)->W.unaryExpr([](const double x) {
                        return pow(x, 2.0);
                    }).sum();
                }

                // calculate cost from mini-batches
                for (T_Size batch = 0, offset = 0; batch < numberOfExamples; batch += batchSize, offset++) {
                    Math::T_Matrix A = this->network.forward(dataSet.getInput(offset, batchSize));
                    Math::T_Matrix Y = dataSet.getOutput(offset, batchSize);

                    T_Size miniBatchSize = (T_Size) Y.cols();

                    double loss = this->network.loss(Y, A); // loss for the mini-batch
                    double error = this->network.error(miniBatchSize); // error for the mini-batch

                    cost += (error * loss + ((this->regularization * penalty) / (2.0 * (double) miniBatchSize)))
                            /
                            // TODO: fix it
                            ((double) numBatches * ((double) miniBatchSize / (double) batchSize));
                }

                Impulse::NeuralNetwork::Trainer::CostGradientResult result;
                result.cost = cost;
                result.gradient = this->network.getRolledGradient();

                return result;
            }

            double AbstractTrainer::accuracy(Impulse::Dataset::SlicedDataset &dataset) {
                T_Size batchSize = 100;
                double result = 0.0;

                // calculate accuracy from mini-batches
                for (T_Size batch = 0, offset = 0; batch < dataset.getNumberOfExamples(); batch += batchSize, offset++) {
                    Math::T_Matrix predictedOutput = this->network.forward(dataset.getInput(offset, batchSize));
                    Math::T_Matrix correctOutput = dataset.getOutput(offset, batchSize);

                    for (T_Size i = 0; i < predictedOutput.cols(); i++) {
                        int index1;
                        int index2;

                        predictedOutput.col(i).maxCoeff(&index1);
                        correctOutput.col(i).maxCoeff(&index2);

                        if (index1 == index2) {
                            result++;
                        }
                    }
                }

                return result / dataset.getNumberOfExamples() * 100;
            }
        }
    }
}