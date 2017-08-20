#include "BaseDatasetModifier.h"

namespace Impulse {

    namespace DatasetModifier {

        BaseDatasetModifier::BaseDatasetModifier(Impulse::Dataset *_dataset) :
                dataset(_dataset) {

        }

        void BaseDatasetModifier::apply(void) {
            Impulse::DatasetData *samples = this->dataset->getSamples();
            if (samples->size() > 0) {
                for (unsigned int i = 0; i < samples->at(0).getSize(); i++) {
                    this->applyToColumn(i);
                }
            }
        }

    }

}
