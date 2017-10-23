#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        namespace Utils {

            Math::T_Matrix im2col(Math::T_Matrix input, int channels,
                                  int height, int width,
                                  int kernel_h, int kernel_w,
                                  int pad_h, int pad_w,
                                  int stride_h, int stride_w) {

                int rows = kernel_w * kernel_h * channels;
                int cols = ((width - kernel_w + 2 * pad_w) / stride_w + 1)
                           *
                           ((height - kernel_h + 2 * pad_h) / stride_h + 1);
                int currentResultCol = 0;

                Math::T_Matrix result(rows, cols);
                result.setZero();

                for (int boundingY = -pad_h;
                     boundingY + kernel_h <= height + 2 * pad_h;
                     boundingY += stride_h) {
                    for (int boundingX = -pad_w;
                         boundingX + kernel_w <= width + 2 * pad_h;
                         boundingX += stride_w) {
                        int currentResultRow = 0;
                        for (int channel = 0; channel < channels; channel++) {
                            int inputOffset = height * width * channel;
                            for (int y = 0; y < kernel_h; y++) {
                                for (int x = 0; x < kernel_w; x++) {
                                    if (boundingY + y >= 0 && boundingX + x >= 0 && boundingX + x < width &&
                                        boundingY + y < height) {

                                        result(currentResultRow, currentResultCol) = input(
                                                ((y + boundingY) * width) + boundingX + x + inputOffset, 0);

                                    }
                                    currentResultRow++;
                                }
                            }
                        }
                        currentResultCol++;
                    }
                }

                return result;
            }
        }
    }
}
