#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            MaxPool::MaxPool() : Abstract() {}

            void MaxPool::configure() {
                this->outputWidth = (this->width - this->filterSize) / this->stride + 1;
                this->outputHeight = (this->height - this->filterSize) / this->stride + 1;
            }

            void MaxPool::setFilterSize(T_Size value) {
                this->filterSize = value;
            }

            void MaxPool::setStride(T_Size value) {
                this->stride = value;
            }

            Math::T_Matrix MaxPool::forward(const Math::T_Matrix &input) {
                Math::T_Matrix result(this->outputWidth * this->outputHeight * this->depth, input.cols());

                // TODO: openmp
#pragma omp parallel
#pragma omp for
                for (T_Size i = 0; i < input.cols(); i++) {
                    Math::T_Matrix maxPool = Utils::maxpool(input.col(i), this->depth,
                                                    this->height, this->width,
                                                    this->filterSize, this->filterSize,
                                                    this->stride, this->stride);
#pragma omp critical
                    {
                        result.col(i) = maxPool;
                    }
                }

                return this->Z = this->A = result;
            }

            Math::T_Matrix MaxPool::activation() {
                return Math::T_Matrix(); // no activation for maxpool layer
            }

            Math::T_Matrix MaxPool::derivative() {
                // TODO
                return Math::T_Matrix();
            }

            const T_String MaxPool::getType() {
                return TYPE_MAXPOOL;
            }

            double MaxPool::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                // TODO
                return 0.0;
            }

            double MaxPool::error(T_Size m) {
                // TODO
                return 0.0;
            }

            void MaxPool::transition(const Layer::LayerPointer &prevLayer) {
                if (prevLayer->getType() == Layer::TYPE_CONV) {
                    this->setSize(prevLayer->getOutputHeight(),
                                  prevLayer->getOutputWidth(),
                                  prevLayer->getOutputDepth());
                }
            }
        }
    }
}
