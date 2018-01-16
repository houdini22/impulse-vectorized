#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagationToMaxPool::BackPropagationToMaxPool
                        (Layer::LayerPointer layer, Layer::LayerPointer previousLayer) :
                        Abstract(layer, previousLayer) {

                }

                Math::T_Matrix BackPropagationToMaxPool::propagate(Math::T_Matrix input,
                                                                   T_Size numberOfExamples,
                                                                   double regularization,
                                                                   Math::T_Matrix delta) {

                    Layer::MaxPool *prevLayer = (Layer::MaxPool *) this->previousLayer.get();
                    Math::T_Matrix result(prevLayer->Z.rows(), prevLayer->Z.cols());
                    result.setZero();

                    T_Size filterSize = prevLayer->getFilterSize();
                    T_Size stride = prevLayer->getStride();
                    T_Size inputWidth = prevLayer->getWidth();
                    T_Size inputHeight = prevLayer->getHeight();
                    T_Size channels = prevLayer->getDepth();
                    T_Size outputWidth = prevLayer->getOutputWidth();
                    T_Size outputHeight = prevLayer->getOutputHeight();

//#pragma omp parallel
//#pragma omp for
                    for (T_Size m = 0; m < numberOfExamples; m++) {
                        for (T_Size channel = 0; channel < channels; channel++) {
                            for (T_Size boundingY = 0, y = 0; boundingY + filterSize <= inputHeight; boundingY += stride, y++) {
                                for (T_Size boundingX = 0, x = 0; boundingX + filterSize <= inputWidth; boundingX += stride, x++) {
                                    double _max = -INFINITY;
                                    T_Size inputOffset = inputHeight * inputWidth * channel;
                                    T_Size outputOffset = outputHeight * outputWidth * channel;
                                    T_Size maxX = 0;
                                    T_Size maxY = 0;

                                    for (T_Size filterY = 0; filterY < filterSize; filterY++) {
                                        for (T_Size filterX = 0; filterX < filterSize; filterX++) {
                                            if (_max < prevLayer->Z(inputOffset + ((filterY + boundingY) * inputWidth) + boundingX + filterX, m)) {
                                                _max = prevLayer->Z(inputOffset + ((filterY + boundingY) * inputWidth) + boundingX + filterX, m);
                                                maxX = filterX;
                                                maxY = filterY;
                                            }
                                        }
                                    }
                                    result(inputOffset + ((maxY + boundingY) * inputWidth) + boundingX + maxX, m) = delta(outputOffset + (y * outputWidth) + x, m);
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