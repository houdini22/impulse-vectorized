#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Pool::Pool() : Abstract() {}

            void Pool::configure() {}

            void Pool::setFilterSize(T_Size value) {
                this->filterSize = value;
            }

            void Pool::setStride(T_Size value) {
                this->stride = value;
            }

            Math::T_Matrix Pool::forward(const Math::T_Matrix &input) {
                this->Z = Utils::maxpool(input, this->depth,
                                         this->height, this->width,
                                         this->filterSize, this->filterSize,
                                         this->stride, this->stride);
                return this->Z;
            }

            Math::T_Matrix Pool::activation() {
                return this->Z;
            }

            Math::T_Matrix Pool::derivative() {
                // TODO
                return Math::T_Matrix();
            }

            const T_String Pool::getType() {
                return TYPE_POOL;
            }

            double Pool::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                // TODO
                return 0.0;
            }

            double Pool::error(T_Size m) {
                // TODO
                return 0.0;
            }

            void Pool::transition(const Layer::LayerPointer &prevLayer) {
                if (prevLayer->getType() == Layer::TYPE_CONV) {
                    this->setSize(prevLayer->getOutputHeight(),
                                  prevLayer->getOutputWidth(),
                                  prevLayer->getOutputDepth());
                }
            }
        }
    }
}
