#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagationToConv::BackPropagationToConv(Layer::LayerPointer layer, Layer::LayerPointer previousLayer) : Abstract(layer, previousLayer) {}

                Math::T_Matrix BackPropagationToConv::propagate(Math::T_Matrix input, T_Size numberOfExamples, double regularization, Math::T_Matrix delta) {

                    auto *previousLayer = (Layer::Conv *) this->previousLayer.get();

                    T_Size padding = previousLayer->getPadding();
                    T_Size stride = previousLayer->getStride();
                    T_Size filterSize = previousLayer->getFilterSize();
                    T_Size outputWidth = previousLayer->getOutputWidth();
                    T_Size outputHeight = previousLayer->getOutputHeight();
                    T_Size outputDepth = previousLayer->getOutputDepth();
                    T_Size inputWidth = previousLayer->getWidth();
                    T_Size inputHeight = previousLayer->getHeight();
                    T_Size inputDepth = previousLayer->getDepth();

                    Math::T_Matrix tmpResult((inputWidth + 2 * padding) * (inputHeight + 2 * padding) * inputDepth, numberOfExamples);
                    tmpResult.setZero();

                    Math::T_Matrix result(inputWidth * inputHeight * inputDepth, numberOfExamples);

                    previousLayer->gW.setZero();
                    previousLayer->gb.setZero();

#pragma omp parallel
#pragma omp for
                    for (T_Size m = 0; m < numberOfExamples; m++) {
                        for (T_Size c = 0; c < outputDepth; c++) {
                            for (T_Size h = 0; h < outputHeight; h++) {
                                for (T_Size w = 0; w < outputWidth; w++) {
                                    T_Size vertStart = stride * h;
                                    T_Size vertEnd = vertStart + filterSize;
                                    T_Size horizStart = stride * w;
                                    T_Size horizEnd = horizStart + filterSize;

                                    // filter loop
                                    for (T_Size d = 0; d < inputDepth; d++) {
                                        for (T_Size y = 0, vertical = vertStart; y < filterSize; y++, vertical++) {
                                            for (T_Size x = 0, horizontal = horizStart; x < filterSize; x++, horizontal++) {
                                                tmpResult(((d * (inputWidth + 2 * padding) * (inputHeight + 2 * padding)) + (vertical * (inputWidth + 2 * padding)) + horizontal), m) +=
                                                        previousLayer->W(c, d * (filterSize * filterSize) + (y * filterSize) + x) *
                                                        delta((c * outputWidth * outputHeight) + (h * outputWidth) + w, m);

                                                double z = 0;
                                                if (padding == 0) {
                                                    z = previousLayer->Z((d * inputWidth * inputHeight) + (vertical * inputWidth) + horizontal, m);
                                                } else {
                                                    if (vertical - padding >= 0 && horizontal - padding >= 0 && vertical - inputHeight &&
                                                        vertical - padding < inputWidth && horizontal - padding < inputHeight) {
                                                        z = previousLayer->Z((d * inputWidth * inputHeight) + ((vertical - padding) * inputWidth) + (horizontal - padding), m);
                                                    }
                                                }

                                                previousLayer->gW(c, d * (filterSize * filterSize) + (y * filterSize) + x) +=
                                                        z *
                                                        delta(c * (outputWidth * outputHeight) + (h * outputWidth) + w, m);
                                            }
                                        }
                                    }

                                    previousLayer->gb(c, 0) += delta(c * (outputWidth * outputHeight) + (h * outputWidth) + w, m);
                                }
                            }
                        }

                        if (padding > 0) { // unpad
                            for (int c = 0; c < inputDepth; c++) {
                                for (int h = -padding, y = 0; h < inputHeight + padding; h++, y++) {
                                    for (int w = -padding, x = 0; w < inputWidth + padding; w++, x++) {
                                        if (w > 0 && h > 0) {
                                            result((inputDepth * inputWidth * inputHeight) + (h * inputWidth) + w, m) =
                                                    tmpResult((inputDepth * (inputWidth + 2 * padding) * (inputHeight + 2 * padding)) + (y * (inputWidth + 2 * padding)) + x, m);
                                        }
                                    }
                                }
                            }
                        }
                    }

                    if (padding > 0) {
                        return result;
                    }

                    return tmpResult;
                }
            }
        }
    }
}