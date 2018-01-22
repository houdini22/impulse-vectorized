#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            namespace Matrix {

                Math::T_Matrix create(T_Index rows, T_Index cols) {
                    Math::T_Matrix m(rows, cols);
                    return m;
                }

                T_Index rows(Math::T_Matrix m) {
                    return m.n_rows;
                }

                T_Index cols(Math::T_Matrix m) {
                    return m.n_cols;
                }

                Math::T_Matrix resize(Math::T_Matrix &m, T_Index rows, T_Index cols) {
                    m.reshape(rows, cols);
                    return m;
                }

                Math::T_Matrix fill(Math::T_Matrix &m, double value) {
                    m.fill(value);
                    return m;
                }

                Math::T_Matrix fillRandom(Math::T_Matrix &m, T_Size i) {
                    m.randu();
                    m.for_each([&i](arma::mat::elem_type &x) {
                        x = (x - 0.5) * 2 * sqrt(2.0 / i);
                    });
                    return m;
                }

                Math::T_Matrix colwiseAdd(Math::T_Matrix m1, Math::T_Matrix m2) {
                    m1.each_col() += m2;
                    return m1;
                }

                Math::T_Matrix subtract(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1 - m2;
                }

                Math::T_Matrix multiply(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1 * m2;
                }

                Math::T_Matrix elementWiseMultiply(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1 % m2;
                }

                Math::T_Matrix divide(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1 / m2;
                }

                Math::T_Matrix pow(Math::T_Matrix m, double i) {
                    return arma::pow(m, i);
                }

                Math::T_Matrix log(Math::T_Matrix m) {
                    return arma::log(m);
                }

                Math::T_Matrix exp(Math::T_Matrix m) {
                    return arma::exp(m);
                }

                double sum(Math::T_Matrix m) {
                    return arma::sum(arma::sum(m));
                }

                Math::T_Matrix colwiseSum(Math::T_Matrix m) {
                    return arma::sum(m, 0);
                }

                Math::T_Matrix rowwiseSum(Math::T_Matrix m) {
                    return arma::sum(m, 1);
                }

                Math::T_Matrix replicateRows(Math::T_Matrix m, T_Index rows) {
                    return arma::repmat(m, rows, 1);
                }

                Math::T_Matrix forEach(Math::T_Matrix m, std::function<double(const double &)> callback) {
                    m.for_each([&callback](arma::mat::elem_type &x) {
                        x = callback(x);
                    });
                    return m;
                }

                Math::T_Matrix conjugate(Math::T_Matrix m) {
                    return arma::conj(m);
                }

                Math::T_Matrix transpose(Math::T_Matrix m) {
                    return m.t();
                }
            }
        }
    }
}
