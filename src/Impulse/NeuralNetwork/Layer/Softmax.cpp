#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Softmax::Softmax() : Abstract1D() {}

            Math::T_Matrix Softmax::activation(Math::T_Matrix m) {
                Math::T_Matrix t = Math::Matrix::exp(m);
                Math::T_Matrix divider = Math::Matrix::replicateRows(Math::Matrix::colwiseSum(t), Math::Matrix::rows(t));
                Math::T_Matrix result = Math::Matrix::divide(t, divider);
                return result;
            }

            Math::T_Matrix Softmax::derivative() {
                static_assert("No derivative for SOFTMAX layer.");
                return Math::T_Matrix();
            }

            const T_String Softmax::getType() {
                return TYPE_SOFTMAX;
            }

            double Softmax::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                Math::T_Matrix loss = Math::Matrix::elementWiseMultiply(output, Math::Matrix::log(predictions));
                return Math::Matrix::sum(loss);
            }

            double Softmax::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
