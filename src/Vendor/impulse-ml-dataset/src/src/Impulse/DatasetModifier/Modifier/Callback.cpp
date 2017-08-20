#include "Callback.h"

namespace Impulse {

    namespace DatasetModifier {

        namespace Modifier {

            void Callback::applyToColumn(int columnIndex) {
                DatasetData *samples = this->dataset->getSamples();
                for (unsigned int i = 0; i < samples->size(); i++) {
                    DatasetSample *sample = &samples->at(i);
                    sample->setColumn(columnIndex, this->callback);
                }
            }

            void Callback::setCallback(std::function<std::string(std::string)> cb) {
                this->callback = cb;
            }

        }

    }

}
