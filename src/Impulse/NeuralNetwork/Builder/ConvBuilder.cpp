#include "../include.h"
#include "../common.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            ConvBuilder::ConvBuilder(T_Dimension dims) : Abstract<Network::ConvNetwork>(dims) {

            }

            void ConvBuilder::firstLayerTransition(Layer::LayerPointer layer) {
                layer->setSize(this->dimension.width,
                               this->dimension.height,
                               this->dimension.depth);
            }
        }
    }
}
