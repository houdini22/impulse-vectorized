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
                auto numberOfExamples = (T_Size) dataSet.getInput().n_cols;
                std::chrono::high_resolution_clock::time_point beginTrain = std::chrono::high_resolution_clock::now();
                double beta1 = this->beta1;
                double beta2 = this->beta2;
                double epsilon = this->epsilon;

                for (T_Size i = 0; i < iterations; i++) {
                    std::chrono::high_resolution_clock::time_point beginIteration = std::chrono::high_resolution_clock::now();

                    for (T_Size batch = 0, offset = 0; batch < numberOfExamples; batch += batchSize, offset++) {
                        std::chrono::high_resolution_clock::time_point beginIterationBatch = std::chrono::high_resolution_clock::now();

                        network.backward(dataSet.getInput(offset, batchSize), dataSet.getOutput(offset, batchSize), network.forward(dataSet.getInput(offset, batchSize)), this->regularization);

                        for (T_Size j = 0; j < network.getSize(); j++) {
                            Layer::LayerPointer layer = network.getLayer(j);

                            if (layer->getType() == Layer::TYPE_MAXPOOL) {
                                continue;
                            }

                            layer->W = layer->W - learningRate * (layer->gW);
                            layer->b = layer->b - learningRate * (layer->gb);
                        }

                        if (this->verbose) {
                            std::chrono::high_resolution_clock::time_point endIterationBatch = std::chrono::high_resolution_clock::now();
                            auto durationBatch = std::chrono::duration_cast<std::chrono::milliseconds>(endIterationBatch - beginIterationBatch).count();
                            std::cout << "Batch: " << (offset + 1) << "/" << ceil((double) numberOfExamples / batchSize)
                                      << " | Time: " << durationBatch << "ms"
                                      << std::endl;
                        }
                    }

                    if (this->verbose) {
                        Trainer::CostGradientResult currentResult = this->cost(dataSet);

                        if ((i + 1) % this->verboseStep == 0) {
                            std::chrono::high_resolution_clock::time_point endIteration = std::chrono::high_resolution_clock::now();
                            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(endIteration - beginIteration).count();
                            std::cout << "Iteration: " << (i + 1)
                                      << " | Cost: " << currentResult.getCost()
                                      << " | Accuracy: " << currentResult.getAccuracy()
                                      << "% | Time: " << duration << "ms"
                                      << std::endl;
                        }
                    }
                }

                if (this->verbose) {
                    std::chrono::high_resolution_clock::time_point endTrain = std::chrono::high_resolution_clock::now();
                    auto duration = std::chrono::duration_cast<std::chrono::seconds>(endTrain - beginTrain).count();
                    std::cout << "Training end. " << duration << "s" << std::endl;
                }
            }
        }
    }
}
