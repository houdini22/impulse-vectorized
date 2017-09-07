#include "Logistic.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            T_Matrix Logistic::activation() {
                return this->Z.unaryExpr([](const double x) {
                    return 1.0 / (1.0 + exp(-x));
                });
            }

            T_Matrix Logistic::derivative() {
                return this->A.array() * (1.0 - this->A.array());
            }

            const T_String Logistic::getType() {
                return TYPE_LOGISTIC;
            }

            double Logistic::loss(T_Matrix output, T_Matrix predictions) {
                T_Matrix loss =
                        (output.array() * predictions.unaryExpr([](const double x) { return log(x); }).array())
                        +
                        (output.unaryExpr([](const double x) { return 1.0 - x; }).array()
                         *
                         predictions.unaryExpr([](const double x) { return log(1.0 - x); }).array()
                        );
                return loss.sum();
            }
        }
    }
}
