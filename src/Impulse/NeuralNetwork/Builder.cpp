#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        Builder::Builder(T_Size inputSize) : network(Network(inputSize)) {
            this->prevSize = inputSize;
        }

        void Builder::createLayer(T_Size size, T_String type) {
            if (type == Layer::TYPE_LOGISTIC) {
                this->network.addLayer(Layer::LayerPointer(new Layer::Logistic(size, this->prevSize)));
            } else if (type == Layer::TYPE_RELU) {
                this->network.addLayer(Layer::LayerPointer(new Layer::Relu(size, this->prevSize)));
            } else if (type == Layer::TYPE_SOFTMAX) {
                this->network.addLayer(Layer::LayerPointer(new Layer::Softmax(size, this->prevSize)));
            } else if (type == Layer::TYPE_PURELIN) {
                this->network.addLayer(Layer::LayerPointer(new Layer::Purelin(size, this->prevSize)));
            }
            this->prevSize = size;
        }

        template<typename LAYER_TYPE>
        void Builder::createLayer(T_Size size, std::function<void(LAYER_TYPE *)> callback) {
            LAYER_TYPE *layer = new LAYER_TYPE(size, this->prevSize);
            Layer::LayerPointer pointer(layer);

            callback(layer);

            this->network.addLayer(pointer);
            this->prevSize = layer->getOutputSize();
        };

        template void Builder::createLayer<Layer::Logistic>(T_Size size, std::function<void(Layer::Logistic *)> callback);
        template void Builder::createLayer<Layer::Softmax>(T_Size size, std::function<void(Layer::Softmax *)> callback);

        template<typename LAYER_TYPE>
        void Builder::createLayer(T_Size size) {
            LAYER_TYPE *layer = new LAYER_TYPE(size, this->prevSize);
            Layer::LayerPointer pointer(layer);

            this->network.addLayer(pointer);
            this->prevSize = layer->getOutputSize();
        };

        template void Builder::createLayer<Layer::Logistic>(T_Size size);
        template void Builder::createLayer<Layer::Softmax>(T_Size size);

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
            for (auto it = savedLayers.begin(); it != savedLayers.end(); ++it) {
                builder.createLayer(it.value()[0], it.value()[1]);
            }

            Math::T_RawVector theta = jsonFile["weights"];
            builder.getNetwork().setRolledTheta(Math::rawToVector(theta));

            return builder;
        }
    }
}