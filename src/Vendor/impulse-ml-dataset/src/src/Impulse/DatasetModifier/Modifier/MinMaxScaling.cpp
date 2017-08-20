#include "MinMaxScaling.h"

namespace Impulse {

    namespace DatasetModifier {

        namespace Modifier {

            void MinMaxScaling::applyToColumn(int columnIndex) {
                DatasetData *samples = this->dataset->getSamples();
                double min = std::numeric_limits<double>::max();
                double max = std::numeric_limits<double>::min();
                unsigned int i;

                for (i = 0; i < samples->size(); i++) {
                    double value = samples->at(i).getColumnToDouble(columnIndex);
                    if (value < min) {
                        min = value;
                    } else if (value > max) {
                        max = value;
                    }
                }

                for (i = 0; i < samples->size(); i++) {
                    double value = samples->at(i).getColumnToDouble(columnIndex);
                    double newValue = (value - min) / (max - min);
                    samples->at(i).setColumn(columnIndex, newValue);
                }
            }

        }

    }

}
