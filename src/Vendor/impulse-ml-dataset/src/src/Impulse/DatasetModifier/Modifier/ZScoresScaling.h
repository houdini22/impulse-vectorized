#ifndef SRC_IMPULSE_DATASETMODIFIER_ZSCORESSCALINGDATASETMODIFIER_H_
#define SRC_IMPULSE_DATASETMODIFIER_ZSCORESSCALINGDATASETMODIFIER_H_

#include <math.h>
#include <vector>
#include <functional>
#include "../../Dataset.h"
#include "../BaseDatasetModifier.h"

namespace Impulse {

    namespace DatasetModifier {

        namespace Modifier {

            class ZScoresScaling : public BaseDatasetModifier {
            public:
                ZScoresScaling(Impulse::Dataset *dataset) :
                        BaseDatasetModifier(dataset) {

                }

                virtual ~ZScoresScaling() {
                }

                void applyToColumn(int columnIndex);

                std::vector<double> applyToSample(std::initializer_list<double> list);
            };

        }

    }

}

#endif /* SRC_IMPULSE_DATASETMODIFIER_ZSCORESSCALINGDATASETMODIFIER_H_ */
