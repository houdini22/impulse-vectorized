#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Softmax::Softmax(T_Size size, T_Size prevSize) : Abstract(size, prevSize) {}

            Math::T_Matrix Softmax::activation() {
                Math::T_Matrix t = this->Z.unaryExpr([](const double x) {
                    return exp(x);
                });
                Math::T_Matrix divider = t.colwise().sum().replicate(t.rows(), 1);
                Math::T_Matrix result = t.array() / divider.array();
                return result;
            }

            Math::T_Matrix Softmax::derivative() {
                // TODO
                return Math::T_Matrix();
            }

            const T_String Softmax::getType() {
                return TYPE_SOFTMAX;
            }

            double Softmax::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                Math::T_Matrix loss = (output.array() *
                                       predictions.unaryExpr([](const double x) { return log(x); }).array());
                return loss.sum();
            }

            double Softmax::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
