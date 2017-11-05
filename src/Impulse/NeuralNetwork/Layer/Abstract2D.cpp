#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract2D::Abstract2D() : Abstract() {}

            void Abstract2D::configure()  {
                // initialize weights
                this->W.resize(this->height, this->width);
                this->W.setRandom();
                this->W = this->W * sqrt(2.0 / this->width);

                // initialize bias
                this->b.resize(this->height);
                this->b.setZero();
            }

            bool Abstract2D::is2d() {
                return true;
            }

            bool Abstract2D::is3d() {
                return false;
            }

            void Abstract2D::transition(const Layer::LayerPointer &prevLayer) {
                if (prevLayer->is2d()) {
                    this->setPrevSize(prevLayer->getSize());
                } else if (prevLayer->is3d()) {
                    this->setPrevSize(prevLayer->getOutputWidth() *
                                      prevLayer->getOutputHeight() *
                                      prevLayer->getOutputDepth());
                }
            }
        }
    }
}