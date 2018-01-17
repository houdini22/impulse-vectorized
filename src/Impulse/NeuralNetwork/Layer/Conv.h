#ifndef IMPULSE_NEURALNETWORK_LAYER_CONV_H
#define IMPULSE_NEURALNETWORK_LAYER_CONV_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            const T_String TYPE_CONV = "conv";

            class Conv : public Abstract3D {
            protected:
                T_Size filterSize = 3;
                T_Size padding = 1;
                T_Size stride = 2;
                T_Size numFilters = 2;
            public:
                Conv();

                void configure() override;

                Math::T_Matrix forward(Math::T_Matrix input) override;

                T_Size getOutputHeight() override;

                T_Size getOutputWidth() override;

                T_Size getOutputDepth() override;

                void setFilterSize(T_Size value);

                T_Size getFilterSize();

                void setPadding(T_Size value);

                T_Size getPadding();

                void setStride(T_Size value);

                T_Size getStride();

                void setNumFilters(T_Size value);

                T_Size getNumFilters();

                Math::T_Matrix activation() override;

                Math::T_Matrix derivative() override;

                const T_String getType() override;

                double loss(Math::T_Matrix output, Math::T_Matrix predictions) override;

                double error(T_Size m) override;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_CONV_H
