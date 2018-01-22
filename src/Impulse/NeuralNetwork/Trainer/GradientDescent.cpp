#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            GradientDescent::GradientDescent(Network::Abstract &net) : AbstractTrainer(net) {}

            void GradientDescent::train(Impulse::Dataset::SlicedDataset &dataSet) {
                Network::Abstract network = this->network;
                double learningRate = this->learningRate;
                T_Size iterations = this->learningIterations;

                for (T_Size i = 0; i < iterations; i++) {
                    std::chrono::high_resolution_clock::time_point begin = std::chrono::high_resolution_clock::now();

                    network.backward(dataSet.getInput(), dataSet.getOutput(), network.forward(dataSet.getInput()), this->regularization);

                    Trainer::CostGradientResult result = this->cost(dataSet);

                    for (T_Size j = 0; j < network.getSize(); j++) {
                        Layer::LayerPointer layer = network.getLayer(j);

                        if (layer->getType() == Layer::TYPE_MAXPOOL) {
                            continue;
                        }

                        layer->W = Math::Matrix::subtract(layer->W, layer->gW * learningRate);
                        layer->b = Math::Matrix::subtract(layer->b, layer->b * learningRate);
                    }

                    Trainer::CostGradientResult currentResult = this->cost(dataSet);

                    if (this->verbose) {
                        if ((i + 1) % this->verboseStep == 0) {
                            std::chrono::high_resolution_clock::time_point end = std::chrono::high_resolution_clock::now();
                            auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count();
                            std::cout << "Iteration: " << (i + 1)
                                      << " | Cost: " << currentResult.getCost()
                                      << " | Accuracy: " << currentResult.getAccuracy()
                                      << "% | Time: " << duration
                                      << std::endl;
                        }
                    }

                    if (currentResult.getCost() > result.getCost()) {
                        std::cout << "Terminated." << std::endl;
                        break;
                    }
                }
            }
        }
    }
}
