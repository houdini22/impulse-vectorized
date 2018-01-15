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

                    T_Size padding = previousLayer->getPadding();
                    T_Size stride = previousLayer->getStride();
                    T_Size filterSize = previousLayer->getFilterSize();
                    T_Size outputWidth = previousLayer->getOutputWidth();
                    T_Size outputHeight = previousLayer->getOutputHeight();
                    T_Size outputDepth = previousLayer->getOutputDepth();
                    T_Size inputWidth = previousLayer->getWidth();
                    T_Size inputHeight = previousLayer->getHeight();
                    T_Size inputDepth = previousLayer->getDepth();

                    Math::T_Matrix tmpResult((inputWidth + padding) * (inputHeight + padding) * inputDepth,
                                             input.cols());
                    tmpResult.setZero();

                    previousLayer->gW.setZero();
                    previousLayer->gb.setZero();

//#pragma omp parallel
//#pragma omp for
                    for (T_Size m = 0; m < numberOfExamples; m++) {
                        for (T_Size c = 0; c < outputDepth; c++) {
                            for (T_Size h = 0; h < outputHeight; h++) {
                                for (T_Size w = 0; w < outputWidth; w++) {
                                    T_Size vertStart = stride * h;
                                    T_Size vertEnd = vertStart + filterSize;
                                    T_Size horizStart = stride * w;
                                    T_Size horizEnd = horizStart + filterSize;

                                    //std::cout << "RESULT: " << c * (outputWidth * outputHeight) + (h * outputWidth) + w << std::endl;

                                    // filter loop
                                    for (T_Size d = 0; d < inputDepth; d++) {
                                        for (T_Size y = 0, vStart = vertStart; y < vertEnd - vertStart; y++, vStart++) {
                                            for (T_Size x = 0, hStart = horizStart;
                                                 x < horizEnd - horizStart; x++, hStart++) {
                                                //previousLayer->W(c, d * (filterSize * filterSize) + (y * filterSize) + x);
                                                //delta(c * (outputWidth * outputHeight) + (h * outputWidth) + w, m);

                                                /*std::cout << "RESULT: "
                                                          << (d * (inputWidth + padding) * (inputHeight + padding)) + (vertStart * (inputWidth + padding)) + horizStart
                                                          << std::endl;*/

                                                tmpResult((d * (inputWidth + padding) * (inputHeight + padding)) +
                                                          (vertStart * (inputWidth + padding)) + horizStart, m) +=
                                                        previousLayer->W(c, d * (filterSize * filterSize) +
                                                                            (y * filterSize) + x) *
                                                        delta(c * (outputWidth * outputHeight) + (h * outputWidth) + w,
                                                              m);

                                                previousLayer->gW(c, d * (filterSize * filterSize) +
                                                                     (y * filterSize) + x) += previousLayer->A(
                                                        (d * (inputWidth + padding) * (inputHeight + padding)) +
                                                        (vertStart * (inputWidth + padding)) + horizStart, m)
                                                                     *
                                                                     delta(c * (outputWidth * outputHeight) +
                                                                           (h * outputWidth) + w,
                                                                           m);
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

                    previousLayer->gb /= numberOfExamples;
                    previousLayer->gW /= numberOfExamples;

                    //std::cout << "CONV DELTA RECEIVED: " << std::endl << delta << std::endl;
                    //std::cout << "CONV DELTA SENT: " << std::endl << tmpResult << std::endl;
                    //std::cout << "DELTA GW: " << std::endl << previousLayer->gW << std::endl;
                    //std::cout << "DELTA GB: " << std::endl << previousLayer->gb << std::endl;

                    return tmpResult;
                }
            }
        }
    }
}