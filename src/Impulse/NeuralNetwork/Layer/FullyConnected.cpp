#include "../include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            FullyConnected::FullyConnected() : Conv() {};

            const T_String FullyConnected::getType() {
                return TYPE_FULLYCONNECTED;
            }

            void FullyConnected::transition(Layer::LayerPointer prevLayer) {
                if (prevLayer->is3D()) {
                    if (prevLayer->getType() == Layer::TYPE_MAXPOOL) {
                        auto layer = (Layer::MaxPool *) prevLayer.get();

                        this->setFilterSize(layer->getOutputWidth());
                        this->setPadding(0);
                        this->setStride(1);
                        this->setWidth(layer->getOutputWidth());
                        this->setHeight(layer->getOutputHeight());
                        this->setDepth(layer->getOutputDepth());
                    } else if (prevLayer->getType() == Layer::TYPE_FULLYCONNECTED) {
                        auto layer = (Layer::FullyConnected *) prevLayer.get();

                        this->setFilterSize(layer->getOutputWidth());
                        this->setPadding(0);
                        this->setStride(1);
                        this->setWidth(layer->getOutputWidth());
                        this->setHeight(layer->getOutputHeight());
                        this->setDepth(layer->getOutputDepth());
                    }
                }
            }

            void FullyConnected::setSize(T_Size value) {
                this->setNumFilters(value);
            }
        }
    }
}
