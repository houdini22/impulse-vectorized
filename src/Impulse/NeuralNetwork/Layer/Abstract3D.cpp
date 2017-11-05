#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract3D::Abstract3D() : Abstract() {}

            bool Abstract3D::is2d() {
                return false;
            }

            bool Abstract3D::is3d() {
                return true;
            }

            void Abstract3D::transition(const Layer::LayerPointer &prevLayer) {
                if (prevLayer->is3d()) {
                    this->setSize(prevLayer->getOutputHeight(),
                                  prevLayer->getOutputWidth(),
                                  prevLayer->getOutputDepth());
                }
            }
        }
    }
}