#ifndef IMPULSE_VECTORIZED_MATRIX_H
#define IMPULSE_VECTORIZED_MATRIX_H

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            namespace Matrix {

                void resize(Math::T_Matrix &m, T_Size rows, T_Size cols);

                void resize(Math::T_Vector &m, T_Size length);

                void fillRandom(Math::T_Matrix &m, T_Size i);

                void fillRandom(Math::T_Vector &m, T_Size i);

                void fill(Math::T_Matrix &m, double i);

                void fill(Math::T_Vector &m, double i);
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_MATRIX_H
