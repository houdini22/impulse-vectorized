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

            Math::T_Matrix Pool::forward(Math::T_Matrix input) {
                std::cout << this->width << "," << this->height << "," << this->depth << std::endl;
                std::cout << "INPUT:" << std::endl << input << std::endl;
                Math::T_Matrix output = Utils::maxpool(input, this->depth, this->height, this->width, this->filterSize, this->filterSize, this->stride, this->stride);
                return output;
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

            T_Size Pool::getOutputSize() {
                return (
                        (this->outputRows * this->outputCols)
                        *
                        this->depth
                );

            }

            double Pool::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                // TODO
                return 0.0;
            }

            double Pool::error(T_Size m) {
                // TODO
                return 0.0;
            }

            void Pool::transition(Layer::LayerPointer prevLayer) {
                if (prevLayer->getType() == Layer::TYPE_CONV) {
                    this->width = prevLayer->getOutputRows();
                    this->height = prevLayer->getOutputCols();
                    this->depth = prevLayer->getDepth();
                }
            }
        }
    }
}
