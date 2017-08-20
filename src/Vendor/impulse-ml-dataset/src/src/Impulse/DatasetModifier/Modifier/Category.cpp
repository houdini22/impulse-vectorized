#include "Category.h"

namespace Impulse {

    namespace DatasetModifier {

        namespace Modifier {

            void Category::applyToColumn(int columnIndex) {
                DatasetData *samples = this->dataset->getSamples();
                std::vector<std::string> categories;
                unsigned int i;

                for (i = 0; i < samples->size(); i++) {
                    DatasetSample *sample = &samples->at(i);
                    std::string category = sample->getColumnToString(columnIndex);
                    if (std::find(categories.begin(), categories.end(), category)
                        == categories.end()) {
                        categories.push_back(category);
                    }
                }

                for (i = 0; i < samples->size(); i++) {
                    DatasetSample *sample = &samples->at(i);
                    std::string existingCategory = sample->getColumnToString(columnIndex);
                    for (unsigned int j = 0; j < categories.size(); j++) {
                        if (categories.at(j) == existingCategory) {
                            sample->insertColumnAfter(columnIndex + j, "1");
                        } else {
                            sample->insertColumnAfter(columnIndex + j, "0");
                        }
                    }
                    sample->removeColumnAt(columnIndex);
                }
            }

        }

    }

}
