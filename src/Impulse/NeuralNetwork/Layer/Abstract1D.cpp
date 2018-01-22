#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Abstract1D::Abstract1D() : Abstract() {}

            void Abstract1D::configure() {
                // initialize weights
                this->W = Math::Matrix::resize(this->W, this->height, this->width);
                this->W = Math::Matrix::fillRandom(this->W, this->width);

                // initialize bias
                this->b = Math::Matrix::resize(this->b, this->height, 1);
                this->b = Math::Matrix::fillRandom(this->b, this->width);

                this->gW = Math::Matrix::resize(this->W, this->height, this->width);
                this->gb = Math::Matrix::resize(this->b, this->height, 1);
            }

            bool Abstract1D::is1D() {
                return true;
            }

            bool Abstract1D::is3D() {
                return false;
            }

            void Abstract1D::transition(Layer::LayerPointer prevLayer) {
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