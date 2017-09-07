#ifndef ABSTRACT_LAYER_H
#define ABSTRACT_LAYER_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../Math/common.h"
#include "../../common.h"

using Impulse::NeuralNetwork::Math::T_Matrix;
using Impulse::NeuralNetwork::Math::T_Vector;
using Impulse::T_Size;

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract {
            protected:
                T_Size size;        // number of neurons
                T_Size prevSize;    // number of prev layer size (input)
            public:
                T_Matrix W;         // weights
                T_Vector b;         // bias
                T_Matrix A;         // output of the layer after activation
                T_Matrix Z;         // output of the layer before activation
                T_Matrix gW;        // gradient for weights
                T_Vector gb;        // gradient for biases

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
                T_Matrix forward(T_Matrix input);

                /**
                 * Calculates activated values.
                 * @param input
                 * @return
                 */
                virtual T_Matrix activation() = 0;

                /**
                 * Calculates derivative. It depends on activation function.
                 * @return
                 */
                virtual T_Matrix derivative() = 0;

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
                virtual double loss(T_Matrix output, T_Matrix predictions) = 0;
            };
        }
    }
}

#endif //ABSTRACT_LAYER_H
