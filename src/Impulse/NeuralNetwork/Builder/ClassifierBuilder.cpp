#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Builder {

            ClassifierBuilder::ClassifierBuilder(T_Dimension dims) : Abstract<Network::ClassifierNetwork>(dims) {
            }

            void ClassifierBuilder::firstLayerTransition(Layer::LayerPointer layer) {
                layer->setPrevSize(this->dimension.width);
            }

            ClassifierBuilder ClassifierBuilder::fromJSON(T_String path) {
                std::ifstream fileStream(path);
                nlohmann::json jsonFile;

                fileStream >> jsonFile;
                fileStream.close();

                std::vector<T_Size> inputSize = jsonFile["inputSize"];

                T_Dimension dimension;
                dimension.width = inputSize.at(0);

                ClassifierBuilder builder(dimension);

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
}
