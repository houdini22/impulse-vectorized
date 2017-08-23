#ifndef IMPULSE_VECTORIZED_ABSTRACTTRAINER_H
#define IMPULSE_VECTORIZED_ABSTRACTTRAINER_H

#include <math.h>
#include "../Network.h"
#include "../../../Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            struct CostGradientResult {
                double error;
                Eigen::VectorXd gradient;

                double &getCost() {
                    return this->error;
                }

                Eigen::VectorXd &getGradient() {
                    return this->gradient;
                }
            };

            class AbstractTrainer {
                Impulse::NeuralNetwork::Network *network;
                double regularization = 0.0;
                unsigned int learningIterations = 1000;
                double learningRate = 0.1;
                bool verbose = true;
            public:
                AbstractTrainer(Impulse::NeuralNetwork::Network *net);

                Impulse::NeuralNetwork::Network *getNetwork();

                //void setRegularization(double regularization);

                void setLearningIterations(unsigned int nb);

                CostGradientResult cost(Impulse::SlicedDataset &dataSet);

                void train(Impulse::SlicedDataset &dataSet);
/*
                virtual double errorForSample(double prediction, double output) = 0;

                virtual double
                calculateOverallError(unsigned int size, double sumErrors, double errorRegularization) = 0;*/
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACTTRAINER_H
