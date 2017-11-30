#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Conv::Conv() : Abstract3D() {}

            void Conv::configure() {
                this->W.resize(this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->W.setOnes(); // this->W.setRandom();

                this->b.resize(this->numFilters, 1);
                this->b.setOnes();
            }

            Math::T_Matrix Conv::forward(const Math::T_Matrix &input) {
                this->Z.resize(this->getOutputWidth() * this->getOutputHeight() * this->getOutputDepth(),
                               input.cols());

#pragma omp parallel
#pragma omp for
                for (T_Size i = 0; i < input.cols(); i++) {
                    Math::T_Matrix conv = Utils::im2col(input.col(i), this->depth,
                                                        this->height, this->width,
                                                        this->filterSize, this->filterSize,
                                                        this->padding, this->padding,
                                                        this->stride, this->stride);

                    Math::T_Matrix tmp = ((this->W * conv).colwise() + this->b).transpose(); // transpose for
                    // rolling to vector
                    Eigen::Map<Math::T_Vector> tmp2(tmp.data(), tmp.size());
                    this->Z.col(i) = tmp2;
                }

                return this->A = this->activation();
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

            void Conv::setPadding(T_Size value) {
                this->padding = value;
            }

            void Conv::setStride(T_Size value) {
                this->stride = value;
            }

            void Conv::setNumFilters(T_Size value) {
                this->numFilters = value;
            }

            Math::T_Matrix Conv::activation() {
                return this->Z.unaryExpr([](const double x) {
                    return std::max(0.0, x); // TODO: set it; relu by default
                });
            }

            Math::T_Matrix Conv::derivative() {
                // TODO
                return Math::T_Matrix();
            }

            const T_String Conv::getType() {
                return TYPE_CONV;
            }

            double Conv::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                // TODO
                return 0.0;
            }

            double Conv::error(T_Size m) {
                // TODO
                return 0.0;
            }

            void Conv::debug() {

            }
        }
    }
}
