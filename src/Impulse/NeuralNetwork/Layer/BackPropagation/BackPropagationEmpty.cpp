#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagationEmpty::BackPropagationEmpty(Layer::LayerPointer layer, Layer::LayerPointer previousLayer) : Abstract(layer, previousLayer) {}

                Math::T_Matrix BackPropagationEmpty::propagate(const Math::T_Matrix &input, T_Size numberOfExamples, double regularization, const Math::T_Matrix &sigma) {
                    return sigma;
                }
            }
        }
    }
}