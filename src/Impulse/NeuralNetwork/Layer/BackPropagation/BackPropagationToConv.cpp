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

                    auto *previousLayer = (Layer::Conv *) this->previousLayer.get();

                    T_Size stride = previousLayer->getStride();
                    T_Size filterSize = previousLayer->getFilterSize();

                    Math::T_Matrix result(
                            previousLayer->getWidth() * previousLayer->getHeight() * previousLayer->getDepth(),
                            input.cols());
                    result.setZero();

                    previousLayer->gW.setZero();
                    previousLayer->gb.setZero();

#pragma omp parallel
#pragma omp for
                    for (T_Size m = 0; m < numberOfExamples; m++) {
                        for (T_Size h = 0; h < previousLayer->getOutputHeight(); h++) {
                            for (T_Size w = 0; w < previousLayer->getOutputWidth(); w++) {
                                for (T_Size c = 0; c < previousLayer->getOutputDepth(); c++) {
                                    T_Size vertStart = stride * h;
                                    T_Size vertEnd = vertStart + filterSize;
                                    T_Size horizStart = stride * w;
                                    T_Size horizEnd = horizStart + filterSize;

                                    for (T_Size x = 0, hStart = horizStart; hStart <= horizEnd; x++, hStart++) {
                                        for (T_Size y = 0, vStart = vertStart; vStart <= vertEnd; y++, vStart++) {
                                            for (T_Size inputChannel = 0;
                                                 inputChannel < previousLayer->getDepth(); inputChannel++) {
                                                result(inputChannel * (vStart * previousLayer->getWidth()) + hStart, m);
                                                previousLayer->W(c, inputChannel * (y * filterSize) + x);
                                            }
                                        }
                                    }
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