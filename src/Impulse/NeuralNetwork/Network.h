#ifndef NETWORK_H
#define NETWORK_H

#include <vector>
#include <iostream>
#include "Layer/Abstract.h"
#include "Math/common.h"
#include "../common.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::NeuralNetwork::Math::T_Vector;
using Impulse::NeuralNetwork::Math::T_RawVector;
using Impulse::NeuralNetwork::Math::rawToVector;
using Impulse::T_Size;
using AbstractLayer = Impulse::NeuralNetwork::Layer::Abstract;

namespace Impulse {

    namespace NeuralNetwork {

        typedef std::vector<AbstractLayer *> LayersContainer;

        class Network {
        protected:
            T_Size size = 0;
            T_Size inputSize = 0;
            LayersContainer layers;
        public:
            Network(T_Size inputSize);

            void addLayer(AbstractLayer *layer);

            T_Matrix forward(T_Matrix input);

            void backward(T_Matrix X, T_Matrix Y, T_Matrix predictions, double regularization);

            T_Size getInputSize();

            T_Size getSize();

            AbstractLayer *getLayer(T_Size key);

            T_Vector getRolledTheta();

            T_Vector getRolledGradient();

            void setRolledTheta(T_Vector theta);
        };
    }
}

#endif //NETWORK_H
