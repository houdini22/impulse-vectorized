#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract1D::Abstract1D() : Abstract() {}

            void Abstract1D::configure() {
                // initialize weights
                Math::Matrix::resize(this->W, this->height, this->width);
                Math::Matrix::fillRandom(this->W, this->width);

                // initialize bias
                Math::Matrix::resize(this->b, this->height);
                Math::Matrix::fillRandom(this->b, this->width);

                // initialize gradient
                Math::Matrix::resize(this->gW, this->height, this->width);
                Math::Matrix::resize(this->gb, this->height);
            }

            bool Abstract1D::is1D() {
                return true;
            }

            bool Abstract1D::is2D() {
                return false;
            }

            void Abstract1D::transition(Layer::LayerPointer prevLayer) {
                if (prevLayer->is1D()) {
                    this->setWidth(prevLayer->getSize());
                } else if (prevLayer->is2D()) {
                    this->setWidth(prevLayer->getOutputWidth() * prevLayer->getOutputHeight() * prevLayer->getOutputDepth());
                }
            }
        }
    }
}