#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Logistic::Logistic() : Abstract1D() {};

            Math::T_Matrix Logistic::activation(Math::T_Matrix m) {
                return Math::Matrix::forEach(m, [](const double x) {
                    return 1.0 / (1.0 + exp(-x));
                });
            }

            Math::T_Matrix Logistic::derivative() {
                return Math::Matrix::elementWiseMultiply(
                        this->A,
                        Math::Matrix::forEach(this->A, [](const double x) {
                            return 1.0 - x;
                        })
                );
            }

            const T_String Logistic::getType() {
                return TYPE_LOGISTIC;
            }

            double Logistic::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                Math::T_Matrix p1 = Math::Matrix::elementWiseMultiply(
                        output,
                        Math::Matrix::forEach(predictions, [](const double x) {
                            return log(x);
                        })
                );
                Math::T_Matrix p2 = Math::Matrix::elementWiseMultiply(
                        Math::Matrix::forEach(output, [](const double x) {
                            return 1.0 - x;
                        }),
                        Math::Matrix::forEach(predictions, [](const double x) {
                            return log(1.0 - x);
                        })
                );
                Math::T_Matrix loss = Math::Matrix::add(p1, p2);
                return Math::Matrix::sum(loss);
            }

            double Logistic::error(T_Size m) {
                return (-1.0 / (double) m);
            }
        }
    }
}
