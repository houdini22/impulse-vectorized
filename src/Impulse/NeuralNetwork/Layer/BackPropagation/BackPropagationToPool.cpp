#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagation1DTo3D::BackPropagation1DTo3D
                        (Layer::LayerPointer layer, Layer::LayerPointer previousLayer) :
                        Abstract(layer, previousLayer) {

                }

                Math::T_Matrix BackPropagation1DTo3D::propagate(Math::T_Matrix input,
                                                                T_Size numberOfExamples,
                                                                double regularization,
                                                                Math::T_Matrix delta) {

                    Math::T_Matrix previousActivations =
                            this->previousLayer == nullptr ? input : this->previousLayer->A;
                    delta = delta * previousActivations.transpose().conjugate();

                    this->layer->gW = delta.array() / numberOfExamples +
                                      (regularization / numberOfExamples * this->layer->W.array());
                    this->layer->gb = delta.rowwise().sum() / numberOfExamples;

                    if (this->previousLayer != nullptr) {
                        Math::T_Matrix tmp1 = this->layer->W.transpose() * delta;
                        Math::T_Matrix tmp2 = previousLayer->derivative();

                        return tmp1.array() * tmp2.array(); // new delta
                    }
                    return Math::T_Matrix(); // return empty - this is first layer
                }
            }
        }
    }
}