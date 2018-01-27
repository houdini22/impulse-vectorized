#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Softmax::Softmax() : Abstract1D() {}

            Math::T_Matrix Softmax::activation(Math::T_Matrix m) {
                Math::T_Matrix t = m.unaryExpr([](const double x) {
                    return exp(x);
                });
                Math::T_Matrix divider = t.colwise().sum().replicate(t.rows(), 1);
                Math::T_Matrix result = t.array() / divider.array();
                return result;
            }

            Math::T_Matrix Softmax::derivative() {
                static_assert(true, "No derivative for SOFTMAX layer.");
                return Math::T_Matrix();
            }

            const T_String Softmax::getType() {
                return TYPE_SOFTMAX;
            }

            double Softmax::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                Math::T_Matrix loss = Math::Matrix::elementWiseMultiply(
                        output,
                        Math::Matrix::forEach(predictions, [](const double x) {
                            return log(x);
                        })
                );
                return Math::Matrix::sum(loss);
            }

            double Softmax::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
