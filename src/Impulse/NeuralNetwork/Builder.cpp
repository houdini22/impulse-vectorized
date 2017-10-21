#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        Builder::Builder(T_Size inputSize) : network(Network(inputSize)) {
            this->prevSize = inputSize;
        }

        template<typename LAYER_TYPE>
        void Builder::createLayer(T_Size size, std::function<void(LAYER_TYPE *)> callback) {
            auto *layer = new LAYER_TYPE(size, this->prevSize);
            Layer::LayerPointer pointer(layer);

            callback(layer);

            layer->configure();

            this->network.addLayer(pointer);
            this->prevSize = layer->getOutputSize();
        };

        template void
        Builder::createLayer<Layer::Logistic>(T_Size size, std::function<void(Layer::Logistic *)> callback);

        template void Builder::createLayer<Layer::Purelin>(T_Size size, std::function<void(Layer::Purelin *)> callback);

        template void Builder::createLayer<Layer::Relu>(T_Size size, std::function<void(Layer::Relu *)> callback);

        template void Builder::createLayer<Layer::Softmax>(T_Size size, std::function<void(Layer::Softmax *)> callback);

        template<typename LAYER_TYPE>
        void Builder::createLayer(T_Size size) {
            auto *layer = new LAYER_TYPE(size, this->prevSize);
            Layer::LayerPointer pointer(layer);

            layer->configure();

            this->network.addLayer(pointer);
            this->prevSize = layer->getOutputSize();
        };

        template void Builder::createLayer<Layer::Logistic>(T_Size size);

        template void Builder::createLayer<Layer::Purelin>(T_Size size);

        template void Builder::createLayer<Layer::Relu>(T_Size size);

        template void Builder::createLayer<Layer::Softmax>(T_Size size);

        template<typename LAYER_TYPE>
        void Builder::createLayer(std::function<void(LAYER_TYPE *)> callback) {
            auto *layer = new LAYER_TYPE();
            Layer::LayerPointer pointer(layer);

            layer->setPrevSize(this->prevSize);
            callback(layer);

            layer->configure();

            this->network.addLayer(pointer);
            this->prevSize = layer->getOutputSize();
        };

        template void Builder::createLayer<Layer::Logistic>(std::function<void(Layer::Logistic *)> callback);

        template void Builder::createLayer<Layer::Purelin>(std::function<void(Layer::Purelin *)> callback);

        template void Builder::createLayer<Layer::Relu>(std::function<void(Layer::Relu *)> callback);

        template void Builder::createLayer<Layer::Softmax>(std::function<void(Layer::Softmax *)> callback);

        Network &Builder::getNetwork() {
            return this->network;
        }

        Builder Builder::fromJSON(T_String path) {
            std::ifstream fileStream(path);
            nlohmann::json jsonFile;

            fileStream >> jsonFile;
            fileStream.close();

            Builder builder((T_Size) jsonFile["inputSize"]);

            nlohmann::json savedLayers = jsonFile["layers"];

            for (auto &element : savedLayers) {
                T_Size size = element[0];
                T_String layerType = element[1];

                Layer::LayerPointer pointer;
                if (layerType == Layer::TYPE_LOGISTIC) {
                    builder.createLayer<Layer::Logistic>(size);
                } else if (layerType == Layer::TYPE_RELU) {
                    builder.createLayer<Layer::Relu>(size);
                } else if (layerType == Layer::TYPE_SOFTMAX) {
                    builder.createLayer<Layer::Softmax>(size);
                } else if (layerType == Layer::TYPE_PURELIN) {
                    builder.createLayer<Layer::Purelin>(size);
                }
            }

            Math::T_RawVector theta = jsonFile["weights"];
            builder.getNetwork().setRolledTheta(Math::rawToVector(theta));

            return builder;
        }
    }
}