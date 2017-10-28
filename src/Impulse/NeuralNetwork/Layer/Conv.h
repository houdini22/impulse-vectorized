#ifndef IMPULSE_NEURALNETWORK_LAYER_CONV_H
#define IMPULSE_NEURALNETWORK_LAYER_CONV_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_CONV = "conv";

            class Conv : public Abstract {
            protected:
                T_Size filterSize = 3;
                T_Size padding = 1;
                T_Size stride = 2;
                T_Size numFilters = 2;
                T_Size outputHeight = 0;
                T_Size outputWidth = 0;
            public:
                Conv();

                void configure() override;

                Math::T_Matrix forward(const Math::T_Matrix &input) override;

                T_Size getOutputHeight() override;

                T_Size getOutputWidth() override;

                T_Size getOutputDepth() override;

                void setFilterSize(T_Size value);

                void setPadding(T_Size value);

                void setStride(T_Size value);

                void setNumFilters(T_Size value);

                Math::T_Matrix activation() override;

                Math::T_Matrix derivative() override;

                const T_String getType() override;

                double loss(Math::T_Matrix output, Math::T_Matrix predictions) override;

                double error(T_Size m) override;

                void debug() override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_CONV_H
