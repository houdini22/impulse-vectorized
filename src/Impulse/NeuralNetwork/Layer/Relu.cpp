#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Relu::Relu() : Abstract1D() {};

            Math::T_Matrix Relu::activation(Math::T_Matrix m) {
                return ActivationFunction::relu(m);
            }

            Math::T_Matrix Relu::derivative() {
                return Derivative::relu(this->A);
            }

            const T_String Relu::getType() {
                return TYPE_RELU;
            }

            double Relu::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                static_assert("No loss function for RELU layer.");
                return 0.0;
            }

            double Relu::error(T_Size m) {
                static_assert("No error function for RELU layer.");
                return 0.0;
            }
        }
    }
}
