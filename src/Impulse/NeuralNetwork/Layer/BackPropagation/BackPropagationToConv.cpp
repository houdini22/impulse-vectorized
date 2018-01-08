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
                    T_Size outputWidth = previousLayer->getOutputWidth();
                    T_Size outputHeight = previousLayer->getOutputHeight();
                    T_Size outputDepth = previousLayer->getOutputDepth();
                    T_Size inputWidth = previousLayer->getWidth();
                    T_Size inputHeight = previousLayer->getHeight();
                    T_Size inputDepth = previousLayer->getDepth();

                    Math::T_Matrix result(inputWidth * inputHeight * inputDepth, input.cols());
                    result.setZero();

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

                                    previousLayer->A(c * (outputWidth * outputHeight) + (h * outputWidth) + w, m);


                                    // filter loop
                                    for (T_Size d = 0; d < inputDepth; d++) {
                                        for (T_Size y = 0, vStart = vertStart; vStart < vertEnd; y++, vStart++) {
                                            for (T_Size x = 0, hStart = horizStart; hStart < horizEnd; x++, hStart++) {
                                                /*previousLayer->W(c, d * (inputWidth * inputHeight) + (filterSize * y) + x);
                                                delta(c * (outputWidth * outputHeight) + (h * outputWidth) + w, m);
                                                result(d * (inputWidth * inputHeight) + (inputWidth * y) + x, m);
                                                std::cout << "DATA: " << x << "," << y << "," << c << std::endl;*/
                                                result(d * (inputWidth * inputHeight) + (h * inputWidth) + w, m) +=
                                                        previousLayer->W(c, d * (filterSize * filterSize) + (filterSize * y) + x) *
                                                                delta(c * (outputWidth * outputHeight) + (h * outputWidth) + w, m);

                                                // std::cout << "RESULT: " << d * (inputWidth * inputHeight) + (h * inputWidth) + w << std::endl;
                                                previousLayer->A(c * (outputWidth * outputHeight), m);
                                            }
                                        }
                                    }


                                    previousLayer->gb(c, 0) += delta(
                                            c * (outputWidth * outputHeight) + (h * outputWidth) + w,
                                            m
                                    );
                                }
                            }
                        }
                    }

                    std::cout << "CONV WEIGHTS SIZE: " << std::endl << previousLayer->W.rows() << ","
                              << previousLayer->W.cols() << std::endl;
                    std::cout << "CONV DELTA RECEIVED: " << std::endl << delta << std::endl;
                    std::cout << "CONV DELTA SENT: " << std::endl << result << std::endl;

                    return result;
                }
            }
        }
    }
}