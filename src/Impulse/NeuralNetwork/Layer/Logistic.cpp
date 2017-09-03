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
        }
    }
}
