#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            template<class NETWORK_TYPE>
            Abstract<NETWORK_TYPE>::Abstract(T_Dimension dims) : network(NETWORK_TYPE(dims)) {
                this->dimension = dims;
            }

            template<class NETWORK_TYPE>
            template<typename LAYER_TYPE>
            void Abstract<NETWORK_TYPE>::createLayer(std::function<void(LAYER_TYPE *)> callback) {
                auto *layer = new LAYER_TYPE();
                Layer::LayerPointer pointer(layer);

                callback(layer);

                if (this->prevLayer != nullptr) {
                    pointer->transition(this->prevLayer);
                } else {
                    this->firstLayerTransition(pointer);
                }

                layer->configure();

                this->network.addLayer(pointer);
                this->prevLayer = pointer;
            };

            template
            class Abstract<Network::ClassifierNetwork>;

            template
            class Abstract<Network::ConvNetwork>;

            template void
            Abstract<Network::ClassifierNetwork>::createLayer<Layer::Logistic>(
                    std::function<void(Layer::Logistic *)> callback);

            template void
            Abstract<Network::ClassifierNetwork>::createLayer<Layer::Purelin>(
                    std::function<void(Layer::Purelin *)> callback);

            template void
            Abstract<Network::ClassifierNetwork>::createLayer<Layer::Relu>(
                    std::function<void(Layer::Relu *)> callback);

            template void
            Abstract<Network::ClassifierNetwork>::createLayer<Layer::Softmax>(
                    std::function<void(Layer::Softmax *)> callback);

            template void
            Abstract<Network::ConvNetwork>::createLayer<Layer::Conv>(
                    std::function<void(Layer::Conv *)> callback);

            template void
            Abstract<Network::ConvNetwork>::createLayer<Layer::Pool>(
                    std::function<void(Layer::Pool *)> callback);

            template<class NETWORK_TYPE>
            NETWORK_TYPE &Abstract<NETWORK_TYPE>::getNetwork() {
                return this->network;
            }
        }
    }
}
