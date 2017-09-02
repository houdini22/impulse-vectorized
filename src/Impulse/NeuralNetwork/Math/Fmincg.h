#ifndef FMINCG_H
#define FMINCG_H

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

#include <functional>
#include <eigen3/Eigen/Core>

#include "../Trainer/AbstractTrainer.h"
#include "../Trainer/CojungateGradientTrainer.h"
#include "../Network.h"
#include "types.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Trainer {

            struct CostGradientResult;
        }

        namespace Math {

            class Fmincg {
            public:
                Fmincg(void) {}

                ~Fmincg(void) {}

                Vector minimize(
                        Trainer::StepFunction stepFunction,
                        Vector theta, unsigned int length, bool verbose);
            };
        }
    }
}

#endif /* FMINCG_H */