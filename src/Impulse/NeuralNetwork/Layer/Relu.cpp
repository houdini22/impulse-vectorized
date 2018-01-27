#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Relu::Relu() : Abstract1D() {};

            Math::T_Matrix Relu::activation(Math::T_Matrix m) {
                return Math::Matrix::forEach(m, [](const double x) {
                    return std::max(0.0, x);
                });
            }

            Math::T_Matrix Relu::derivative() {
                return Math::Matrix::forEach(this->A, [](const double x) {
                    if (x > 0.0) {
                        return 1.0;
                    }
                    return 0.0;
                });
            }

            const T_String Relu::getType() {
                return TYPE_RELU;
            }

            double Relu::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                static_assert(true, "No loss for RELU layer.");
                return 0.0;
            }

            double Relu::error(T_Size m) {
                static_assert(true, "No error for RELU layer.");
                return 0.0;
            }
        }
    }
}
