#include "common.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            T_RawVector vectorToRaw(T_Vector &vec) {
                return T_RawVector(vec.data(), vec.data() + vec.rows() * vec.cols());
            }

            T_Vector rawToVector(T_RawVector &vec) {
                return Eigen::Map<T_Vector, Eigen::Unaligned>(vec.data(), vec.size());
            }
        }
    }
}
