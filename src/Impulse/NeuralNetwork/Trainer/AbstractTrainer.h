#ifndef ABSTRACT_TRAINER_H
#define ABSTRACT_TRAINER_H

#include <math.h>
#include "../Network.h"
#include "../Math/common.h"
#include "common.h"
#include "../../common.h"
#include "../../../Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

using Impulse::NeuralNetwork::Math::T_Vector;
using Impulse::NeuralNetwork::Network;
using Impulse::T_Size;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

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

#endif //ABSTRACT_TRAINER_H
