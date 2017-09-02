#ifndef MATH_TYPES_H
#define MATH_TYPES_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include <vector>

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            typedef Eigen::MatrixXd T_Matrix;
            typedef Eigen::VectorXd T_Vector;
            typedef std::vector<double> T_RawVector;

            T_RawVector vectorToRaw(T_Vector &vec);
            T_Vector rawToVector(T_RawVector &vec);
        }
    }
}

#endif /* MATH_TYPES_H */
