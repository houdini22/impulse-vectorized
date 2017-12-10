#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagationPointer Factory::create(Layer::LayerPointer layer, Layer::LayerPointer previousLayer) {
                    if (previousLayer == nullptr) {
                        if (layer->is1D()) {
                            return BackPropagationPointer(new BackPropagation1DTo1D(layer, previousLayer));
                        }
                    } else {
                        if (previousLayer->getType() == Layer::TYPE_CONV) {
                            return BackPropagationPointer(new BackPropagationToPool(layer, previousLayer));
                        } else if (previousLayer->is1D() && layer->is1D()) {
                            return BackPropagationPointer(new BackPropagation1DTo1D(layer, previousLayer));
                        }
                    }
                    return nullptr;
                }
            }
        }
    }
}