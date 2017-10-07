#ifndef IMPULSE_NEURALNETWORK_MATH_FMINCG_H
#define IMPULSE_NEURALNETWORK_MATH_FMINCG_H

// number of extrapolation runs, set to a higher value for smaller ravine landscapes
#define EXT 3.0
// a bunch of constants for line searches
#define RHO 0.01
// RHO and SIG are the constants in the Wolfe-Powell conditions
#define SIG 0.5
// don't reevaluate within 0.1 of the limit of the current bracket
#define INT 0.1
// max 20 function evaluations per line search
#define MAX 20
// maximum allowed slope ratio
#define RATIO 100.0

#include "../include.h"

using namespace std::chrono;
using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Math {

            class Fmincg {
            public:
                Fmincg() = default;

                ~Fmincg() = default;

                /**
                 * Minimizes multivariate function.
                 * @param stepFunction
                 * @param theta
                 * @param length
                 * @param verbose
                 * @return
                 */
                Math::T_Vector minimize(
                        Trainer::StepFunction stepFunction,
                        Math::T_Vector theta,
                        T_Size length,
                        bool verbose
                );
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_MATH_FMINCG_H