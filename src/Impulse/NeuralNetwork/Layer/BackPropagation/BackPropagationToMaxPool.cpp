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

                    Layer::MaxPool *layer = (Layer::MaxPool *) this->previousLayer.get();
                    Math::T_Matrix result(layer->Z.rows(), layer->Z.cols());
                    result.setZero();

                    T_Size filterSize = layer->getFilterSize();
                    T_Size stride = layer->getStride();
                    T_Size width = layer->getWidth();
                    T_Size height = layer->getHeight();
                    T_Size channels = layer->getDepth();

#pragma omp parallel
#pragma omp for
                    for (T_Size i = 0; i < numberOfExamples; i++) {

                        for (int boundingY = 0;
                             boundingY + filterSize <= height;
                             boundingY += stride) {
                            for (int boundingX = 0;
                                 boundingX + filterSize <= width;
                                 boundingX += stride) {
                                for (int channel = 0; channel < channels; channel++) {
                                    double _max = -INFINITY;
                                    int inputOffset = height * width * channel;
                                    int maxX = 0;
                                    int maxY = 0;
                                    for (int y = 0; y < filterSize; y++) {
                                        for (int x = 0; x < filterSize; x++) {
                                            if (_max <
                                                layer->Z(inputOffset + ((y + boundingY) * width) + boundingX + x, i)) {
                                                _max = layer->Z(inputOffset + ((y + boundingY) * width) + boundingX + x,
                                                                i);
                                                maxX = x;
                                                maxY = y;
                                            }
                                        }
                                    }
                                    result(inputOffset + ((maxY + boundingY) * width) + boundingX + maxX, i) = 1;
                                }
                            }
                        }
                    }

                    std::cout << "MAX POOL DELTA RECEIVED: " << std::endl << delta << std::endl;
                    std::cout << "MAX POOL DELTA SENT: " << std::endl << result << std::endl;

                    return result;
                }
            }
        }
    }
}