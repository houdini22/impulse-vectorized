#ifndef IMPULSE_NEURALNETWORK_LAYER_ABSTRACT_H
#define IMPULSE_NEURALNETWORK_LAYER_ABSTRACT_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            /**
             * Layer pointer.
             */
            typedef std::shared_ptr<Layer::Abstract> LayerPointer;

            class Abstract {
            protected:
                T_Size size;        // number of neurons
                T_Size prevSize;    // number of prev layer size (input)
            public:
                Math::T_Matrix W;         // weights
                Math::T_Vector b;         // bias
                Math::T_Matrix A;         // output of the layer after activation
                Math::T_Matrix Z;         // output of the layer before activation
                Math::T_Matrix gW;        // gradient for weights
                Math::T_Vector gb;        // gradient for biases

                /**
                 * Constructor.
                 * @param size
                 * @param prevSize
                 */
                Abstract(T_Size size, T_Size prevSize);

                /**
                 * Forward propagation.
                 * @param input
                 * @return
                 */
                Math::T_Matrix forward(Math::T_Matrix input);

                /**
                 * Calculates activated values.
                 * @param input
                 * @return
                 */
                virtual Math::T_Matrix activation() = 0;

                /**
                 * Calculates derivative. It depends on activation function.
                 * @return
                 */
                virtual Math::T_Matrix derivative() = 0;

                /**
                 * Getter for layer type.
                 * @return
                 */
                virtual const T_String getType() = 0;

                /**
                 * Getter for layer size.
                 * @return
                 */
                T_Size getSize();

                /**
                 * Loss for the last, classifier layer.
                 * @param output
                 * @param predictions
                 * @return
                 */
                virtual double loss(Math::T_Matrix output, Math::T_Matrix predictions) = 0;

                /**
                 * Error term for network.
                 * @param m
                 * @return
                 */
                virtual double error(T_Size m) = 0;
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_ABSTRACT_H
