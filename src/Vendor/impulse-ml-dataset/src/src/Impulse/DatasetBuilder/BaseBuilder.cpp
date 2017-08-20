#include "../DatasetBuilder/BaseBuilder.h"

namespace Impulse {

    namespace DatasetBuilder {

        Impulse::DatasetSample BaseBuilder::createSample(std::vector<std::string> vec) {
            Impulse::DatasetSample sample(vec);
            return sample;
        }

    }

}
