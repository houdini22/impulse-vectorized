#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagation1DTo1D::BackPropagation1DTo1D(Layer::LayerPointer layer, Layer::LayerPointer previousLayer) : Abstract(layer, previousLayer) {}

                Math::T_Matrix BackPropagation1DTo1D::propagate(const Math::T_Matrix &input,
                                                                T_Size numberOfExamples,
                                                                double regularization,
                                                                const Math::T_Matrix &sigma) {

                    Math::T_Matrix previousActivations = this->previousLayer == nullptr ? input : this->previousLayer->A;
                    Math::T_Matrix delta = Math::Matrix::multiply(sigma, Math::Matrix::conjugate(Math::Matrix::transpose(previousActivations)));

                    this->layer->gW = (delta / numberOfExamples) + (regularization / numberOfExamples * this->layer->W);
                    this->layer->gb = Math::Matrix::rowwiseSum(sigma) / numberOfExamples;

                    if (this->previousLayer != nullptr) {
                        Math::T_Matrix tmp1 = Math::Matrix::multiply(Math::Matrix::transpose(this->layer->W), sigma);
                        Math::T_Matrix tmp2 = this->previousLayer->derivative();

                        return Math::Matrix::elementWiseMultiply(tmp1, tmp2);
                    }
                    return Math::T_Matrix(); // return empty - this is first layer
                }
            }
        }
    }
}