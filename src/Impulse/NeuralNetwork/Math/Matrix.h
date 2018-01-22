#ifndef IMPULSE_VECTORIZED_MATRIX_H
#define IMPULSE_VECTORIZED_MATRIX_H

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            namespace Matrix {

                typedef unsigned long long T_Index;

                inline Math::T_Matrix create(T_Index rows, T_Index cols) {
                    Math::T_Matrix m(rows, cols);
                    return m;
                }

                inline T_Index rows(Math::T_Matrix m) {
                    return m.n_rows;
                }

                inline T_Index cols(Math::T_Matrix m) {
                    return m.n_cols;
                }

                inline Math::T_Matrix resize(Math::T_Matrix m, T_Index rows, T_Index cols) {
                    m.reshape(rows, cols);
                    return m;
                }

                inline Math::T_Matrix fill(Math::T_Matrix m, double value) {
                    m.fill(value);
                    return m;
                }

                inline Math::T_Matrix fillRandom(Math::T_Matrix m, T_Size i) {
                    m.randu();
                    m.for_each([&i](arma::mat::elem_type &x) {
                        x = (x - 0.5) * 2 * sqrt(1.0 / i);
                    });
                    return m;
                }

                inline Math::T_Matrix colwiseAdd(Math::T_Matrix m1, Math::T_Matrix m2) {
                    m1.each_col() += m2;
                    return m1;
                }

                inline Math::T_Matrix multiply(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1 * m2;
                }

                inline Math::T_Matrix elementWiseMultiply(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1 % m2;
                }

                inline Math::T_Matrix divide(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1 / m2;
                }

                inline Math::T_Matrix pow(Math::T_Matrix m, double i) {
                    return arma::pow(m, i);
                }

                inline Math::T_Matrix log(Math::T_Matrix m) {
                    return arma::log(m);
                }

                inline Math::T_Matrix exp(Math::T_Matrix m) {
                    return arma::exp(m);
                }

                inline double sum(Math::T_Matrix m) {
                    return arma::sum(arma::sum(m));
                }

                inline Math::T_Matrix colwiseSum(Math::T_Matrix m) {
                    return arma::sum(m, 0);
                }

                inline Math::T_Matrix replicateRows(Math::T_Matrix m, T_Index rows) {
                    return arma::repmat(m, rows, 1);
                }

                inline Math::T_Matrix forEach(Math::T_Matrix m, std::function<double(const double&)> callback) {
                    m.for_each([&callback](arma::mat::elem_type &x) {
                        x = callback(x);
                    });
                    return m;
                }
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_MATRIX_H
