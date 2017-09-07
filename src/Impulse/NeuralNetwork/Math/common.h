#ifndef IMPULSE_NEURALNETWORK_MATH_COMMON_H
#define IMPULSE_NEURALNETWORK_MATH_COMMON_H

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

#endif //IMPULSE_NEURALNETWORK_MATH_COMMON_H
