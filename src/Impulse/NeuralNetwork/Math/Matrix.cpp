#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            namespace Matrix {

                Math::T_Matrix create(T_Size rows, T_Size cols) {
                    Math::T_Matrix ret(rows, cols);
                    return ret;
                }

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

                Math::T_Matrix elementWiseSubtract(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1.array() - m2.array();
                }

                Math::T_Matrix multiply(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1 * m2;
                }

                Math::T_Matrix elementWiseMultiply(Math::T_Matrix m1, Math::T_Matrix m2) {
                    return m1.array() * m2.array();
                }

                Math::T_Matrix add(Math::T_Matrix m, Math::T_Vector v) {
                    return m.colwise() + v;
                }

                Math::T_Matrix forEach(Math::T_Matrix m, std::function<double(const double)> callback) {
                    return m.unaryExpr([&callback](const double x) {
                        return callback(x);
                    });
                }

                double sum(Math::T_Matrix m) {
                    return m.sum();
                }

                Math::T_Vector rollToVector(Math::T_Matrix m) {
                    Eigen::Map<Math::T_Vector> v(m.transpose().data(), m.size());
                    return v;
                }
            }
        }
    }
}