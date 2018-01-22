#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Logistic::Logistic() : Abstract1D() {};

            Math::T_Matrix Logistic::activation(Math::T_Matrix m) {
                return ActivationFunction::logistic(m);
            }

            Math::T_Matrix Logistic::derivative() {
                return Derivative::logistic(this->A);
            }

            const T_String Logistic::getType() {
                return TYPE_LOGISTIC;
            }

            double Logistic::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                Math::T_Matrix p1 = Math::Matrix::log(predictions);
                Math::T_Matrix p2 = Math::Matrix::forEach(predictions, [](const double &x) {
                    return log(1.0 - x);
                });

                Math::T_Matrix output2 = Math::Matrix::forEach(output, [](const double &x) {
                    return 1.0 - x;
                });

                Math::T_Matrix loss = Math::Matrix::elementWiseMultiply(output, p1) + Math::Matrix::elementWiseMultiply(output2, p2);
                return arma::sum(arma::sum(loss));
            }

            double Logistic::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
