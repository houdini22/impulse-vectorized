#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Conv::Conv() : Abstract3D() {}

            void Conv::configure() {
                this->W.resize(this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->W.setRandom();
                this->W = this->W.array() * sqrt(2.0 / (this->width * this->height * this->depth));

                this->b.resize(this->numFilters, 1);
                this->b.setRandom();
                this->b = this->b.array() * sqrt(2.0 / (this->width * this->height * this->depth));

                this->gW.resize(this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->gb.resize(this->numFilters, 1);
            }

            Math::T_Matrix Conv::forward(Math::T_Matrix input) {
                this->Z.resize(this->width * this->height * this->depth, input.cols());
                this->A.resize(this->getOutputWidth() * this->getOutputHeight() * this->getOutputDepth(),
                               input.cols());

#pragma omp parallel
#pragma omp for
                for (T_Size i = 0; i < input.cols(); i++) {
                    this->Z.col(i) = input.col(i);

                    Math::T_Matrix conv = Utils::im2col(input.col(i), this->depth,
                                                        this->height, this->width,
                                                        this->filterSize, this->filterSize,
                                                        this->padding, this->padding,
                                                        this->stride, this->stride);

                    //std::cout << "CONV: " << conv.rows() << "," << conv.cols() << std::endl << conv << std::endl;

                    Math::T_Matrix tmp = ((this->W * conv).colwise() + this->b).transpose(); // transpose for
                    // rolling to vector
                    Eigen::Map<Math::T_Vector> tmp2(tmp.data(), tmp.size());
                    this->A.col(i) = tmp2;

                }
                this->A = this->activation();

                //std::cout << "CONV INPUT: " << input.rows() << "," << input.cols() << std::endl << input << std::endl;
                //std::cout << "CONV ACTIVATED: " << this->A.rows() << "," << this->A.cols() << std::endl << this->A << std::endl;

                // normalization
                //this->A.colwise().normalize();
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

            Math::T_Matrix Conv::activation() {
                return this->A.unaryExpr([](const double x) {
                    return std::max(0.0, x); // TODO: set it; relu by default
                });
            }

            Math::T_Matrix Conv::derivative() {
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
                std::cout << this->getOutputWidth() << "," << this->getOutputHeight() << "," << this->getOutputDepth()
                          << std::endl;
            }
        }
    }
}
