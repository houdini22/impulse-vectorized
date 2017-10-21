#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Purelin::Purelin() : Abstract() {};

            Purelin::Purelin(T_Size size, T_Size prevSize) : Abstract(size, prevSize) {};

            Math::T_Matrix Purelin::activation() {
                return this->Z;
            }

            Math::T_Matrix Purelin::derivative() {
                Math::T_Matrix d(this->A.rows(), this->A.cols());
                d.setOnes();
                return d;
            }

            const T_String Purelin::getType() {
                return TYPE_PURELIN;
            }

            double Purelin::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                Math::T_Matrix loss = (predictions.array() - output.array()).unaryExpr([](const double x) {
                    return pow(x, 2.0);
                });
                return loss.sum();
            }

            double Purelin::error(T_Size m) {
                return (1.0 / (2.0 * (double) m));
            }
        }
    }
}
