#ifndef IMPULSE_VECTORIZED_MATRIX_H
#define IMPULSE_VECTORIZED_MATRIX_H

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            namespace Matrix {

                Math::T_Matrix create(T_Size rows, T_Size cols);

                void resize(Math::T_Matrix &m, T_Size rows, T_Size cols);

                void resize(Math::T_Vector &m, T_Size length);

                void fillRandom(Math::T_Matrix &m, T_Size i);

                void fillRandom(Math::T_Vector &m, T_Size i);

                void fill(Math::T_Matrix &m, double i);

                void fill(Math::T_Vector &m, double i);

                Math::T_Matrix elementWiseSubtract(Math::T_Matrix m1, Math::T_Matrix m2);

                Math::T_Matrix multiply(Math::T_Matrix m1, Math::T_Matrix m2);

                Math::T_Matrix elementWiseMultiply(Math::T_Matrix m1, Math::T_Matrix m2);

                Math::T_Matrix add(Math::T_Matrix m, Math::T_Vector v);

                Math::T_Matrix forEach(Math::T_Matrix m, std::function<double(const double)> callback);

                double sum(Math::T_Matrix m);

                Math::T_Vector rollToVector(Math::T_Matrix m);
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_MATRIX_H