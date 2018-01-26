#ifndef IMPULSE_VECTORIZED_MATRIX_H
#define IMPULSE_VECTORIZED_MATRIX_H

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            namespace Matrix {

                typedef unsigned long long T_Index;

                Math::T_Matrix create(T_Index rows, T_Index cols);

                T_Index rows(Math::T_Matrix m);

                T_Index cols(Math::T_Matrix m);

                Math::T_Matrix resize(Math::T_Matrix &m, T_Index rows, T_Index cols);

                Math::T_Matrix fill(Math::T_Matrix &m, double value);

                Math::T_Matrix fillRandom(Math::T_Matrix &m, T_Size i);

                Math::T_Matrix colwiseAdd(Math::T_Matrix m1, Math::T_Matrix m2);

                Math::T_Matrix subtract(Math::T_Matrix m1, Math::T_Matrix m2);

                Math::T_Matrix multiply(Math::T_Matrix m1, Math::T_Matrix m2);

                Math::T_Matrix elementWiseMultiply(Math::T_Matrix m1, Math::T_Matrix m2);

                Math::T_Matrix divide(Math::T_Matrix m1, Math::T_Matrix m2);

                Math::T_Matrix pow(Math::T_Matrix m, double i);

                Math::T_Matrix log(Math::T_Matrix m);

                Math::T_Matrix exp(Math::T_Matrix m);

                double sum(Math::T_Matrix m);

                Math::T_Matrix colwiseSum(Math::T_Matrix m);

                Math::T_Matrix rowwiseSum(Math::T_Matrix m);

                Math::T_Matrix replicateRows(Math::T_Matrix m, T_Index rows);

                Math::T_Matrix forEach(Math::T_Matrix m, std::function<double(const double &)> callback);

                Math::T_Matrix conjugate(Math::T_Matrix m);

                Math::T_Matrix transpose(Math::T_Matrix m);

                Math::T_ColVector toVector(Math::T_Matrix m);
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_MATRIX_H
