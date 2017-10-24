#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Relu::Relu() : Abstract() {};

            Math::T_Matrix Relu::activation() {
                return this->Z.unaryExpr([](const double x) {
                    return std::max(0.0, x);
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

            double Relu::error(T_Size m) {
                return 0.0; // TODO
            }
        }
    }
}
