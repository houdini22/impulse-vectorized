#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Conv::Conv() : Abstract() {}

            void Conv::configure() {
                this->W.resize(this->numFilters, this->filterSize * this->filterSize * this->depth);
                this->W.setRandom();

                this->b.resize(this->numFilters, 1);
                this->b.setOnes();

                this->outputWidth = (this->width - this->filterSize + 2 * this->padding) / this->stride + 1;
                this->outputHeight = (this->height - this->filterSize + 2 * this->padding) / this->stride + 1;

                this->Z.resize(this->numFilters, this->outputWidth * this->outputHeight);
                this->Z.setZero();
            }

            T_Size Conv::getOutputHeight() {
                return this->outputWidth;
            }

            T_Size Conv::getOutputWidth() {
                return this->outputHeight;
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
                return this->Z;
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
