#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            namespace Matrix {

                void resize(Math::T_Matrix &m, T_Size rows, T_Size cols) {
                    m.resize(rows, cols);
                }

                void resize(Math::T_Vector &m, T_Size length) {
                    m.resize(length);
                }

                void fillRandom(Math::T_Matrix &m, T_Size i) {
                    m.setRandom();
                    m = m.unaryExpr([&i](const double x) {
                        return x * sqrt(2.0 / i);
                    });
                }

                void fillRandom(Math::T_Vector &m, T_Size i) {
                    m.setRandom();
                    m = m.unaryExpr([&i](const double x) {
                        return x * sqrt(2.0 / i);
                    });
                }

                void fill(Math::T_Matrix &m, double i) {
                    m = m.unaryExpr([&i](const double x) {
                        return i;
                    });
                }

                void fill(Math::T_Vector &m, double i) {
                    m = m.unaryExpr([&i](const double x) {
                        return i;
                    });
                }
            }
        }
    }
}