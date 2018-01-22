#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            Purelin::Purelin() : Abstract1D() {};

            Math::T_Matrix Purelin::activation(Math::T_Matrix m) {
                return m;
            }

            Math::T_Matrix Purelin::derivative() {
                Math::T_Matrix d = Math::Matrix::create(Math::Matrix::rows(this->Z), Math::Matrix::cols(this->Z));
                Math::Matrix::fill(d, 1.0);
                return d;
            }

            const T_String Purelin::getType() {
                return TYPE_PURELIN;
            }

            double Purelin::loss(Math::T_Matrix output, Math::T_Matrix predictions) {
                Math::T_Matrix loss = Math::Matrix::pow(Math::Matrix::subtract(predictions, output), 2.0);
                return Math::Matrix::sum(loss);
            }

            double Purelin::error(T_Size m) {
                return (1.0 / (2.0 * (double) m));
            }
        }
    }
}
