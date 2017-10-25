#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        Builder::Builder(T_Size inputSize) : network(Network(inputSize)) {
            this->inputSize = inputSize;
        }

        template<typename LAYER_TYPE>
        void Builder::createLayer(std::function<void(LAYER_TYPE *)> callback) {
            auto *layer = new LAYER_TYPE();
            Layer::LayerPointer pointer(layer);

            callback(layer);

            if (this->prevLayer != nullptr) {
                layer->transition(this->prevLayer);
            } else {
                layer->setWidth(7);
                layer->setHeight(7);
                layer->setDepth(3);
                // layer->setPrevSize(this->inputSize);
            }

            layer->configure();

            this->network.addLayer(pointer);
            this->prevLayer = pointer;
        };

        template void Builder::createLayer<Layer::Logistic>(std::function<void(Layer::Logistic *)> callback);

        template void Builder::createLayer<Layer::Purelin>(std::function<void(Layer::Purelin *)> callback);

        template void Builder::createLayer<Layer::Relu>(std::function<void(Layer::Relu *)> callback);

        template void Builder::createLayer<Layer::Softmax>(std::function<void(Layer::Softmax *)> callback);

        template void Builder::createLayer<Layer::Conv>(std::function<void(Layer::Conv *)> callback);

        template void Builder::createLayer<Layer::Pool>(std::function<void(Layer::Pool *)> callback);

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
                    builder.createLayer<Layer::Logistic>([&size](auto *layer) {
                        layer->setSize(size);
                    });
                } else if (layerType == Layer::TYPE_RELU) {
                    builder.createLayer<Layer::Relu>([&size](auto *layer) {
                        layer->setSize(size);
                    });
                } else if (layerType == Layer::TYPE_SOFTMAX) {
                    builder.createLayer<Layer::Softmax>([&size](auto *layer) {
                        layer->setSize(size);
                    });
                } else if (layerType == Layer::TYPE_PURELIN) {
                    builder.createLayer<Layer::Purelin>([&size](auto *layer) {
                        layer->setSize(size);
                    });
                }
            }

            Math::T_RawVector theta = jsonFile["weights"];
            builder.getNetwork().setRolledTheta(Math::rawToVector(theta));

            return builder;
        }
    }
}