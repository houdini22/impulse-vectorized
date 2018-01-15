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

//#pragma omp parallel
//#pragma omp for
                    for (T_Size i = 0; i < numberOfExamples; i++) {
                        for (int boundingY = 0, y = 0;
                             boundingY + filterSize <= height;
                             boundingY += stride, y++) {
                            for (int boundingX = 0, x = 0;
                                 boundingX + filterSize <= width;
                                 boundingX += stride, x++) {
                                for (int channel = 0; channel < channels; channel++) {
                                    double _max = -INFINITY;
                                    int inputOffset = height * width * channel;
                                    int maxX = 0;
                                    int maxY = 0;
                                    for (int filterY = 0; filterY < filterSize; filterY++) {
                                        for (int filterX = 0; filterX < filterSize; filterX++) {
                                            if (_max < layer->Z(inputOffset + ((filterY + boundingY) * width) + boundingX + filterX, i)) {
                                                _max = layer->Z(inputOffset + ((filterY + boundingY) * width) + boundingX + filterX, i);
                                                maxX = filterX;
                                                maxY = filterY;
                                            }
                                        }
                                    }
                                    result(inputOffset + ((maxY + boundingY) * width) + boundingX + maxX, i) = delta(inputOffset + ((maxY + boundingY) * width) + boundingX + maxX, i);
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