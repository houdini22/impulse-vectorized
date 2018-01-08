#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            MaxPool::MaxPool() : Abstract3D() {}

            void MaxPool::configure() {}

            void MaxPool::setFilterSize(T_Size value) {
                this->filterSize = value;
            }

            T_Size MaxPool::getFilterSize() {
                return this->filterSize;
            }

            void MaxPool::setStride(T_Size value) {
                this->stride = value;
            }

            T_Size MaxPool::getStride() {
                return this->stride;
            }

            Math::T_Matrix MaxPool::forward(const Math::T_Matrix &input) {
                this->Z = input;
                Math::T_Matrix result(this->getOutputWidth() *
                                      this->getOutputHeight() *
                                      this->getOutputDepth(), input.cols());

#pragma omp parallel
#pragma omp for
                for (T_Size i = 0; i < input.cols(); i++) {
                    Math::T_Matrix maxPool = Utils::maxpool(input.col(i), this->depth,
                                                            this->height, this->width,
                                                            this->filterSize, this->filterSize,
                                                            this->stride, this->stride);
                    result.col(i) = maxPool;
                }

                std::cout << "MAX POOL INPUT: " << input.rows() << "," << input.cols() << std::endl << input << std::endl;
                std::cout << "MAX POOL OUTPUT: " << result.rows() << "," << result.cols() << std::endl << result << std::endl;

                return this->A = result;
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

            T_Size MaxPool::getOutputHeight() {
                return (this->height - this->filterSize) / this->stride + 1;;
            }

            T_Size MaxPool::getOutputWidth() {
                return (this->width - this->filterSize) / this->stride + 1;
            }

            T_Size MaxPool::getOutputDepth() {
                return this->depth;
            }

            void MaxPool::debug() {
                std::cout << this->getOutputWidth() << "," << this->getOutputHeight() << "," << this->getOutputDepth()
                          << std::endl;
            }
        }
    }
}
