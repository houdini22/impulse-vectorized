#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Relu::Relu(T_Size size, T_Size prevSize) : Abstract(size, prevSize) {}

            Math::T_Matrix Relu::activation() {
                return this->Z.unaryExpr([](const double x) {
                    if (x < 0.0) {
                        return 0.0;
                    }
                    return x;
                });
            }

            Math::T_Matrix Relu::derivative() {
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

            double Relu::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                // TODO
                return 0.0;
            }
        }
    }
}
