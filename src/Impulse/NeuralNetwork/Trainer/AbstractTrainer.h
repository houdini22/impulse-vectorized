#ifndef IMPULSE_VECTORIZED_ABSTRACTTRAINER_H
#define IMPULSE_VECTORIZED_ABSTRACTTRAINER_H

#include <math.h>
#include "../Network.h"
#include "../Math/Matrix.h"
#include "../../types.h"
#include "../../../Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

using Vector = Impulse::NeuralNetwork::Math::T_Vector;
using Impulse::NeuralNetwork::Network;
using Impulse::T_Size;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            struct CostGradientResult {
                double error;
                Vector gradient;
                double getError() {
                    return this->error;
                }
                Vector getGradient() {
                    return this->gradient;
                }
            };

            class AbstractTrainer {
            protected:
                Network *network;
                double regularization = 0.0;
                T_Size learningIterations = 1000;
                double learningRate = 0.1;
                bool verbose = true;
                int verboseStep = 100;
            public:
                AbstractTrainer(Network *net);

                Network *getNetwork();

                void setRegularization(double value);

                void setLearningIterations(T_Size value);

                void setLearningRate(double value);

                void setVerbose(bool value);

                void setVerboseStep(int value);

                CostGradientResult cost(Impulse::SlicedDataset &dataSet);

                virtual void train(Impulse::SlicedDataset &dataSet) = 0;
            };

        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACTTRAINER_H
