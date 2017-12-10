#include "../../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            namespace BackPropagation {

                BackPropagationPointer Factory::create(Layer::LayerPointer layer, Layer::LayerPointer previousLayer) {
                    if (previousLayer == nullptr) {
                        if (layer->is2d()) {
                            return BackPropagationPointer(new BackPropagation1DTo1D(layer, previousLayer));
                        }
                    } else {
                        if (previousLayer->is2d() && layer->is2d()) {
                            return BackPropagationPointer(new BackPropagation1DTo1D(layer, previousLayer));
                        }
                    }
                    return nullptr;
                }
            }
        }
    }
}