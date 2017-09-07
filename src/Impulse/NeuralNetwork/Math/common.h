#ifndef MATH_TYPES_H
#define MATH_TYPES_H

#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            typedef Eigen::MatrixXd T_Matrix;
            typedef Eigen::VectorXd T_Vector;
            typedef std::vector<double> T_RawVector;

            Math::T_RawVector vectorToRaw(Math::T_Vector &vec);

            Math::T_Vector rawToVector(Math::T_RawVector &vec);
        }
    }
}

#endif /* MATH_TYPES_H */
