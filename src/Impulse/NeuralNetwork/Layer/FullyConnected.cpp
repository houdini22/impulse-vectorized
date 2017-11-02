#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            FullyConnected::FullyConnected() : Logistic() {};

            void FullyConnected::transition(const Layer::LayerPointer &prevLayer) {
                if (prevLayer->getType() == Layer::TYPE_MAXPOOL) {
                    this->setPrevSize(
                            prevLayer->getOutputWidth() * prevLayer->getOutputHeight() * prevLayer->getOutputDepth());
                } else if (prevLayer->getType() == Layer::TYPE_FULLYCONNECTED) {
                    this->setPrevSize(prevLayer->getSize());
                }
            }

            const T_String FullyConnected::getType() {
                return TYPE_FULLYCONNECTED;
            }
        }
    }
}
