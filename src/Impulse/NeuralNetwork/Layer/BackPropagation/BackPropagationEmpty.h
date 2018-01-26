#ifndef IMPULSE_VECTORIZED_BACKPROPAGATION3DTO1D_H
#define IMPULSE_VECTORIZED_BACKPROPAGATION3DTO1D_H

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                class BackPropagationEmpty : public Abstract {
                public:
                    BackPropagationEmpty(Layer::LayerPointer layer, Layer::LayerPointer previousLayer);

                    Math::T_Matrix propagate(const Math::T_Matrix &input, T_Size numberOfExamples, double regularization, const Math::T_Matrix &sigma);
                };
            }
        }
    }
}

#endif //IMPULSE_VECTORIZED_BACKPROPAGATION3DTO1D_H
