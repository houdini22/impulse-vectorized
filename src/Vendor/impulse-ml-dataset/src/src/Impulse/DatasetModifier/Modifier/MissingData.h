#ifndef SRC_IMPULSE_DATASETMODIFIER_MISSINGDATADATASETMODIFIER_H_
#define SRC_IMPULSE_DATASETMODIFIER_MISSINGDATADATASETMODIFIER_H_

#include <string>
#include "../BaseDatasetModifier.h"

namespace Impulse {

    namespace DatasetModifier {

        namespace Modifier {

            class MissingData : public BaseDatasetModifier {
            protected:
                std::string modificationType = "mean";
            public:
                MissingData(Impulse::Dataset *dataset) :
                        BaseDatasetModifier(dataset) {

                }

                void setModificationType(std::string type);

                void applyToColumn(int columnIndex);
            };

        }

    }

}

#endif /* SRC_IMPULSE_DATASETMODIFIER_MISSINGDATADATASETMODIFIER_H_ */
