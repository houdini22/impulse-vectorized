#include "Builder.h"

namespace Impulse {

    namespace NeuralNetwork {

        Builder::Builder(T_Size inputSize) {
            this->network = new Network(inputSize);
            this->prevSize = inputSize;
        }

        void Builder::createLayer(T_Size size, T_String type) {
            if (type == Layer::TYPE_LOGISTIC) {
                this->network->addLayer(new Layer::Logistic(size, this->prevSize));
            } else if (type == Layer::TYPE_RELU) {
                this->network->addLayer(new Layer::Relu(size, this->prevSize));
            }
            this->prevSize = size;
        }

        Network *Builder::getNetwork() {
            return this->network;
        }

        Builder Builder::fromJSON(T_String path) {
            std::ifstream fileStream(path);
            json jsonFile;

            fileStream >> jsonFile;
            fileStream.close();

            Builder builder((T_Size) jsonFile["inputSize"]);

            json savedLayers = jsonFile["layers"];
            for (auto it = savedLayers.begin(); it != savedLayers.end(); ++it) {
                builder.createLayer(it.value()[0], it.value()[1]);
            }

            T_RawVector theta = jsonFile["weights"];
            builder.getNetwork()->setRolledTheta(Math::rawToVector(theta));

            return builder;
        }
    }
}