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
                    T_Size width = prevLayer->getWidth();
                    T_Size height = prevLayer->getHeight();
                    T_Size channels = prevLayer->getDepth();

//#pragma omp parallel
//#pragma omp for
                    for (T_Size m = 0; m < numberOfExamples; m++) {
                        for (T_Size channel = 0; channel < channels; channel++) {
                            for (T_Size boundingY = 0; boundingY + filterSize <= height; boundingY += stride) {
                                for (T_Size boundingX = 0; boundingX + filterSize <= width; boundingX += stride) {
                                    double _max = -INFINITY;
                                    T_Size inputOffset = height * width * channel;
                                    T_Size maxX = 0;
                                    T_Size maxY = 0;

                                    for (T_Size filterY = 0; filterY < filterSize; filterY++) {
                                        for (T_Size filterX = 0; filterX < filterSize; filterX++) {
                                            if (_max < prevLayer->Z(inputOffset + ((filterY + boundingY) * width) + boundingX + filterX, m)) {
                                                _max = prevLayer->Z(inputOffset + ((filterY + boundingY) * width) + boundingX + filterX, m);
                                                maxX = filterX;
                                                maxY = filterY;
                                            }
                                        }
                                    }
                                    result(inputOffset + ((maxY + boundingY) * width) + boundingX + maxX, m) = delta(inputOffset + ((maxY + boundingY) * width) + boundingX + maxX, m);
                                }
                            }
                        }
                    }

                    //std::cout << "MAX POOL DELTA RECEIVED: " << std::endl << delta << std::endl;
                    //std::cout << "MAX POOL DELTA SENT: " << std::endl << result << std::endl;

                    return result;
                }
            }
        }
    }
}