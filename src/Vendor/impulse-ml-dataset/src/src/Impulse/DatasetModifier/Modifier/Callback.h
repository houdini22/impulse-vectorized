#ifndef SRC_IMPULSE_DATASETMODIFIER_CALLBACKDATASETMODIFIER_H_
#define SRC_IMPULSE_DATASETMODIFIER_CALLBACKDATASETMODIFIER_H_

#include <functional>
#include <string>
#include <iostream>
#include "../../Dataset.h"
#include "../../DatasetSample.h"
#include "../BaseDatasetModifier.h"

namespace Impulse {

    namespace DatasetModifier {

        namespace Modifier {

            class Callback : public BaseDatasetModifier {
            protected:
                std::function<std::string(std::string)> callback;
            public:
                Callback(Impulse::Dataset *dataset) :
                        BaseDatasetModifier(dataset) {
                }

                void setCallback(std::function<std::string(std::string)> cb);

                void applyToColumn(int columnIndex);

            };

        }

    }

}

#endif /* SRC_IMPULSE_DATASETMODIFIER_CALLBACKDATASETMODIFIER_H_ */
