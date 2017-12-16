#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagationToConv::BackPropagationToConv
                        (Layer::LayerPointer layer, Layer::LayerPointer previousLayer) :
                        Abstract(layer, previousLayer) {

                }

                Math::T_Matrix BackPropagationToConv::propagate(Math::T_Matrix input,
                                                                T_Size numberOfExamples,
                                                                double regularization,
                                                                Math::T_Matrix delta) {

                    Layer::LayerPointer previousLayer = this->previousLayer;
                    Math::T_Matrix result(previousLayer->A.rows(), previousLayer->A.cols());
                    result.setZero();

                    this->layer->gW.setZero();
                    this->layer->gb.setZero();

#pragma omp parallel
#pragma omp for
                    for (T_Size i = 0; i < numberOfExamples; i++) {
                        for (T_Size y = 0; y < this->previousLayer->getHeight(); y++) {
                            for (T_Size x = 0; x < this->previousLayer->getWidth(); x++) {
                                for (T_Size c = 0; c < this->previousLayer->getDepth(); c++) {

                                }
                            }
                        }
                    }

                    return result;
                }
            }
        }
    }
}