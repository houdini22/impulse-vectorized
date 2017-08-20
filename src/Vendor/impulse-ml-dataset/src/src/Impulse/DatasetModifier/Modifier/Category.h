#ifndef SRC_IMPULSE_DATASETMODIFIER_CATEGORYDATASETMODIFIER_H_
#define SRC_IMPULSE_DATASETMODIFIER_CATEGORYDATASETMODIFIER_H_

#include <algorithm>
#include "../../Dataset.h"
#include "../../DatasetSample.h"
#include "../BaseDatasetModifier.h"

namespace Impulse {

    namespace DatasetModifier {

        namespace Modifier {

            class Category : public BaseDatasetModifier {
            public:
                Category(Impulse::Dataset *dataset) :
                        BaseDatasetModifier(dataset) {

                }

                void applyToColumn(int columnIndex);
            };

        }

    }

}

#endif /* SRC_IMPULSE_DATASETMODIFIER_CATEGORYDATASETMODIFIER_H_ */
