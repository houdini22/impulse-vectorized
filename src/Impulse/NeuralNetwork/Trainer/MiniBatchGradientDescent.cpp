#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            MiniBatchGradientDescent::MiniBatchGradientDescent(Network::Abstract &net) : AbstractTrainer(net) {}

            void MiniBatchGradientDescent::setBatchSize(T_Size value) {
                this->batchSize = value;
            }

            void MiniBatchGradientDescent::train(Impulse::Dataset::SlicedDataset &dataSet) {
                Network::Abstract network = this->network;
                double learningRate = this->learningRate;
                T_Size iterations = this->learningIterations;
                T_Size batchSize = this->batchSize;
                T_Size numberOfExamples = Math::Matrix::cols(dataSet.getInput());

                auto beginTrain = Utils::timestamp();

                for (T_Size i = 0; i < iterations; i++) {
                    auto beginIteration = Utils::timestamp();

                    for (T_Size batch = 0, offset = 0; batch < numberOfExamples; batch += batchSize, offset++) {
                        auto beginIterationBatch = Utils::timestamp();

                        network.backward(dataSet.getInput(offset, batchSize), dataSet.getOutput(offset, batchSize), network.forward(dataSet.getInput(offset, batchSize)), this->regularization);

                        for (T_Size j = 0; j < network.getSize(); j++) {
                            Layer::LayerPointer layer = network.getLayer(j);

                            if (layer->getType() == Layer::TYPE_MAXPOOL) {
                                continue;
                            }

                            layer->W = Math::Matrix::subtract(layer->W, layer->gW * learningRate);
                            layer->b = Math::Matrix::subtract(layer->b, layer->b * learningRate);
                        }

                        if (this->verbose) {
                            auto endIterationBatch = Utils::timestamp();
                            auto durationBatch = Utils::timeDifferenceMS(endIterationBatch, beginIterationBatch);
                            std::cout << "Batch: " << (offset + 1) << "/" << ceil((double) numberOfExamples / batchSize)
                                      << " | Time: " << durationBatch << "ms"
                                      << std::endl;
                        }
                    }

                    if (this->verbose) {
                        Trainer::CostGradientResult currentResult = this->cost(dataSet);

                        if ((i + 1) % this->verboseStep == 0) {
                            auto endIteration = Utils::timestamp();
                            auto duration = Utils::timeDifferenceMS(endIteration, beginIteration);
                            std::cout << "Iteration: " << (i + 1)
                                      << " | Cost: " << currentResult.getCost()
                                      << " | Accuracy: " << currentResult.getAccuracy()
                                      << "% | Time: " << duration << "ms"
                                      << std::endl;
                        }
                    }
                }

                if (this->verbose) {
                    auto endTrain = Utils::timestamp();
                    auto duration = Utils::timeDifferenceS(endTrain, beginTrain);
                    std::cout << "Training end. " << duration << "s" << std::endl;
                }
            }
        }
    }
}
