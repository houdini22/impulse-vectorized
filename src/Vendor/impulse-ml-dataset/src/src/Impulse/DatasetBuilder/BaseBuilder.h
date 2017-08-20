#ifndef SRC_IMPULSE_DATASET_BASEBUILDER_H_
#define SRC_IMPULSE_DATASET_BASEBUILDER_H_

#include <vector>
#include <string>
#include "../DatasetSample.h"
#include "../Dataset.h"

namespace Impulse {

    namespace DatasetBuilder {

        class BaseBuilder {
        protected:
            Impulse::Dataset dataset;

            Impulse::DatasetSample createSample(std::vector<std::string> vec);

        public:
            virtual ~BaseBuilder() {
            }

            virtual Impulse::Dataset build() = 0;
        };

    }

}

#endif /* SRC_IMPULSE_DATASET_BASEBUILDER_H_ */
