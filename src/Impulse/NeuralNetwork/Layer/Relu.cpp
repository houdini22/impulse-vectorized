#include "Relu.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            T_Matrix Relu::activation() {
                return this->Z.unaryExpr([](const double x) {
                    if (x < 0.0) {
                        return 0.0;
                    }
                    return x;
                });
            }

            T_Matrix Relu::derivative() {
                return this->A.unaryExpr([](const double x) {
                    if (x < 0.0) {
                        return 0.0;
                    }
                    return 1.0;
                });
            }

            const T_String Relu::getType() {
                return TYPE_RELU;
            }

            double Relu::loss(T_Matrix output, T_Matrix predictions) {
                // TODO
                return 0.0;
            }
        }
    }
}
