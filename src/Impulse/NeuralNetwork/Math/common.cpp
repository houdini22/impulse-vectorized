#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            Math::T_RawVector vectorToRaw(Math::T_ColVector &vec) {
                return Math::T_RawVector(vec.memptr(), vec.memptr() + Math::Matrix::rows(vec) * Math::Matrix::cols(vec));
            }

            Math::T_ColVector rawToVector(Math::T_RawVector &vec) {
                //return Eigen::Map<Math::T_Vector, Eigen::Unaligned>(vec.data(), vec.size());
            }
        }
    }
}
