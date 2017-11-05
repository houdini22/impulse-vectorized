#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            FullyConnected::FullyConnected() : Logistic() {};

            const T_String FullyConnected::getType() {
                return TYPE_FULLYCONNECTED;
            }
        }
    }
}
