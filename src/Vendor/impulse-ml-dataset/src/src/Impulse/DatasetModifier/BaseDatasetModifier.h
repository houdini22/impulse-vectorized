#ifndef SRC_IMPULSE_DATASETMODIFIER_BASEDATASETMODIFIER_H_
#define SRC_IMPULSE_DATASETMODIFIER_BASEDATASETMODIFIER_H_

#include <string>
#include <functional>
#include <map>
#include "../Dataset.h"

namespace Impulse {

    namespace DatasetModifier {

        class BaseDatasetModifier {
        protected:
            Impulse::Dataset *dataset;
            std::map<std::string, std::map<double, std::string>> data;
        public:
            BaseDatasetModifier(Impulse::Dataset *dataset);

            virtual ~BaseDatasetModifier() {
            }

            void apply(void);

            virtual void applyToColumn(int columnIndex) = 0;
        };

    }

}

#endif /* SRC_IMPULSE_DATASETMODIFIER_BASEDATASETMODIFIER_H_ */
