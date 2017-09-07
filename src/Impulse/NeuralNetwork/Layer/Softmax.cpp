#include "Softmax.h"
#include <iostream>

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            T_Matrix Softmax::activation() {
                T_Matrix t = this->Z.unaryExpr([](const double x) {
                    return exp(x);
                });
                T_Matrix divider = t.colwise().sum().replicate(t.rows(), 1);
                T_Matrix result = t.array() / divider.array();
                return result;
            }

            T_Matrix Softmax::derivative() {
                // TODO
                return T_Matrix();
            }

            const T_String Softmax::getType() {
                return TYPE_SOFTMAX;
            }

            double Softmax::loss(T_Matrix output, T_Matrix predictions) {
                T_Matrix loss = (output.array() * predictions.unaryExpr([](const double x) { return log(x); }).array());
                return loss.sum();
            }
        }
    }
}
