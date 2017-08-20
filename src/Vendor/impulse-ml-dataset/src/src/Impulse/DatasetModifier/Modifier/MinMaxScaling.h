#ifndef SRC_IMPULSE_DATASETMODIFIER_MINMAXSCALING_H_
#define SRC_IMPULSE_DATASETMODIFIER_MINMAXSCALING_H_

#include <limits>
#include "../BaseDatasetModifier.h"
#include "../../Dataset.h"

namespace Impulse {

    namespace DatasetModifier {

        namespace Modifier {

            class MinMaxScaling : public BaseDatasetModifier {
            public:
                MinMaxScaling(Impulse::Dataset *dataset) :
                        BaseDatasetModifier(dataset) {

                }

                void applyToColumn(int columnIndex);
            };

        }

    }

}

#endif /* SRC_IMPULSE_DATASETMODIFIER_MINMAXSCALING_H_ */
