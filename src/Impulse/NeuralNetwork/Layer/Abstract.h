#ifndef IMPULSE_NEURALNETWORK_LAYER_ABSTRACT_H
#define IMPULSE_NEURALNETWORK_LAYER_ABSTRACT_H

#include "../include.h"

using namespace Impulse::NeuralNetwork;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            // fwd declaration
            class Abstract;

            /**
             * Layer pointer.
             */
            typedef std::shared_ptr<Abstract> LayerPointer;

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
                 * Pure constructor
                 */
                Abstract();

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
                virtual Math::T_Matrix forward(const Math::T_Matrix &input);

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
                 * Setter for size.
                 * @param value
                 */
                void setSize(T_Size value);

                /**
                 * Setter for prev size.
                 * @param value
                 */
                void setPrevSize(T_Size value);

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

                /**
                 * Gets output size.
                 * @return
                 */
                virtual T_Size getOutputSize();

                /**
                 * Finish configuration of the layer
                 */
                virtual void configure();

                /**
                 * Debug.
                 */
                virtual void debug() {};

                /**
                 * Transition
                 */
                virtual void transition(Layer::LayerPointer prevLayer);

                /**
                 * Get output Rows
                 */
                virtual T_Size getOutputRows();

                /**
                 * Get output Cols
                 */
                virtual T_Size getOutputCols();

                /**
                 * Get depth
                 */
                virtual T_Size getDepth();
            };
        }
    }
}

#endif //IMPULSE_NEURALNETWORK_LAYER_ABSTRACT_H
