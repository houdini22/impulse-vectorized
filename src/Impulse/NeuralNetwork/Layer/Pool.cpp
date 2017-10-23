#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Pool::Pool() : Abstract() {}

            void Pool::configure() {
                this->outputRows = (this->width - this->filterSize) / this->stride + 1;
                this->outputCols = (this->height - this->filterSize) / this->stride + 1;
            }

            void Pool::setFilterSize(T_Size value) {
                this->filterSize = value;
            }

            void Pool::setStride(T_Size value) {
                this->stride = value;
            }

            Math::T_Matrix Pool::forward(Math::T_Matrix input) {

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
        }
    }
}
