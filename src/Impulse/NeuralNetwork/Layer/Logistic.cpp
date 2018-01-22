#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Logistic::Logistic() : Abstract1D() {};

            Math::T_Matrix Logistic::activation(Math::T_Matrix m) {
                return ActivationFunction::logisticActivation(m);
            }

            Math::T_Matrix Logistic::derivative() {
                return Derivative::logisticDerivative(this->A);
            }

            const T_String Logistic::getType() {
                return TYPE_LOGISTIC;
            }

            double Logistic::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                Math::T_Matrix p1(predictions);
                p1 = arma::log(p1);

                Math::T_Matrix p2(predictions);
                p2.for_each([](arma::mat::elem_type &x) { x = log(1.0 - x); });

                Math::T_Matrix output2(output);
                output2.for_each([](arma::mat::elem_type &x) { x = 1.0 - x; });

                Math::T_Matrix loss = (output % p1) + (output2 % p2);
                return arma::sum(arma::sum(loss));
            }

            double Logistic::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
