#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract2D::Abstract2D() : Abstract() {}

            bool Abstract2D::is1D() {
                return false;
            }

            bool Abstract2D::is2D() {
                return true;
            }

            void Abstract2D::transition(Layer::LayerPointer prevLayer) {
                if (prevLayer->is2D()) {
                    this->setSize(prevLayer->getOutputHeight(), prevLayer->getOutputWidth(), prevLayer->getOutputDepth());
                }
            }
        }
    }
}