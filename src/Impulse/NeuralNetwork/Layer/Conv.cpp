#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Conv::Conv() : Abstract3D() {}

            void Conv::configure() {
                Math::Matrix::resize(this->W, this->numFilters, this->filterSize * this->filterSize * this->depth);
                Math::Matrix::fillRandom(this->W, this->width * this->height * this->depth);

                Math::Matrix::resize(this->b, this->numFilters);
                Math::Matrix::fill(this->b, 0.01);

                Math::Matrix::resize(this->gW, this->numFilters, this->filterSize * this->filterSize * this->depth);
                Math::Matrix::resize(this->gb, this->numFilters);
            }

            Math::T_Matrix Conv::forward(const Math::T_Matrix &input) {
                this->Z = input;

                Math::T_Matrix result = Math::Matrix::create(this->getOutputWidth() * this->getOutputHeight() * this->getOutputDepth(), (T_Size) input.cols());

#pragma omp parallel
#pragma omp for
                for (T_Size i = 0; i < input.cols(); i++) {
                    Math::T_Matrix conv = Utils::im2col(input.col(i), this->depth,
                                                        this->height, this->width,
                                                        this->filterSize, this->filterSize,
                                                        this->padding, this->padding,
                                                        this->stride, this->stride);

                    Math::T_Matrix tmp = Math::Matrix::add(
                            Math::Matrix::multiply(this->W, conv),
                            this->b
                    );
                    result.col(i) = Math::Matrix::rollToVector(tmp);
                }

                this->A = this->activation(result);
                return this->A;
            }

            T_Size Conv::getOutputHeight() {
                return (this->width - this->filterSize + 2 * this->padding) / this->stride + 1;
            }

            T_Size Conv::getOutputWidth() {
                return (this->height - this->filterSize + 2 * this->padding) / this->stride + 1;
            }

            T_Size Conv::getOutputDepth() {
                return this->numFilters;
            }

            void Conv::setFilterSize(T_Size value) {
                this->filterSize = value;
            }

            T_Size Conv::getFilterSize() {
                return this->filterSize;
            }

            void Conv::setPadding(T_Size value) {
                this->padding = value;
            }

            T_Size Conv::getPadding() {
                return this->padding;
            }

            void Conv::setStride(T_Size value) {
                this->stride = value;
            }

            T_Size Conv::getStride() {
                return this->stride;
            }

            void Conv::setNumFilters(T_Size value) {
                this->numFilters = value;
            }

            T_Size Conv::getNumFilters() {
                return this->numFilters;
            }

            Math::T_Matrix Conv::activation(Math::T_Matrix m) {
                return Math::Matrix::forEach(m, [](const double x) {
                    return std::max(0.0, x);
                });
            }

            Math::T_Matrix Conv::derivative() {
                return Math::Matrix::forEach(this->A, [](const double x) {
                    if (x > 0.0) {
                        return 1.0;
                    }
                    return 0.0;
                });
            }

            const T_String Conv::getType() {
                return TYPE_CONV;
            }

            double Conv::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                static_assert(true, "No loss for CONV layer.");
                return 0.0;
            }

            double Conv::error(T_Size m) {
                static_assert(true, "No error for CONV layer.");
                return 0.0;
            }
        }
    }
}
