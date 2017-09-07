#ifndef IMPULSE_NEURALNETWORK_NETWORK_H
#define IMPULSE_NEURALNETWORK_NETWORK_H

#include "include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        typedef std::vector<Layer::Abstract *> LayersContainer;

        class Network {
        protected:
            T_Size size = 0;
            T_Size inputSize = 0;
            LayersContainer layers;
        public:
            Network(T_Size inputSize);

            void addLayer(Layer::Abstract *layer);

            Math::T_Matrix forward(Math::T_Matrix input);

            void backward(Math::T_Matrix X, Math::T_Matrix Y, Math::T_Matrix predictions, double regularization);

            T_Size getInputSize();

            T_Size getSize();

            Layer::Abstract *getLayer(T_Size key);

            Math::T_Vector getRolledTheta();

            Math::T_Vector getRolledGradient();

            void setRolledTheta(Math::T_Vector theta);

            double loss(Math::T_Matrix output, Math::T_Matrix predictions);
        };
    }
}

#endif //IMPULSE_NEURALNETWORK_NETWORK_H
