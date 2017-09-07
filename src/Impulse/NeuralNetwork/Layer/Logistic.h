#ifndef LOGISTIC_LAYER_H
#define LOGISTIC_LAYER_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_LOGISTIC = "logistic";

            class Logistic : public Abstract {
            protected:
            public:
                Logistic(T_Size size, T_Size prevSize) : Abstract(size, prevSize) {};

                Math::T_Matrix activation();

                Math::T_Matrix derivative();

                const T_String getType();

                double loss(Math::T_Matrix output, Math::T_Matrix predictions);
            };
        }
    }
}

#endif //LOGISTIC_LAYER_H
