#ifndef SRC_IMPULSE_DATASETSPLITTER_H_
#define SRC_IMPULSE_DATASETSPLITTER_H_

#include "DatasetSlicer.h"
#include "../Dataset.h"

namespace Impulse {

    struct SplittedDataset {
        Impulse::SlicedDataset primary;
        Impulse::SlicedDataset secondary;
    };

    namespace DatasetModifier {

        class DatasetSplitter {
        protected:
            Impulse::SlicedDataset *slicedDataset;
        public:
            DatasetSplitter(Impulse::SlicedDataset *slicedDataset);

            Impulse::SplittedDataset split(double ratio);
        };

    }

}

#endif /* SRC_IMPULSE_DATASETSPLITTER_H_ */
