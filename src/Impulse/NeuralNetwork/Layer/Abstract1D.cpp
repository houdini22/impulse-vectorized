#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract1D::Abstract1D() : Abstract() {}

            void Abstract1D::configure() {
                // initialize weights
                this->W.resize(this->height, this->width);
                this->W.setOnes();
                //this->W.setRandom();
                //this->W = this->W * sqrt(2.0 / this->width);

                // initialize bias
                this->b.resize(this->height);
                this->b.setOnes();
            }

            bool Abstract1D::is1D() {
                return true;
            }

            bool Abstract1D::is3D() {
                return false;
            }

            void Abstract1D::transition(const Layer::LayerPointer &prevLayer) {
                if (prevLayer->is1D()) {
                    this->setPrevSize(prevLayer->getSize());
                } else if (prevLayer->is3D()) {
                    this->setPrevSize(prevLayer->getOutputWidth() *
                                      prevLayer->getOutputHeight() *
                                      prevLayer->getOutputDepth());
                }
            }
        }
    }
}