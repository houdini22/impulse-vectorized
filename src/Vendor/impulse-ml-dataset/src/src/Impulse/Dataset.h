#ifndef SRC_IMPULSE_DATASET_H_
#define SRC_IMPULSE_DATASET_H_

#include <vector>
#include <iostream>
#include "DatasetSample.h"

namespace Impulse {

    typedef std::vector<Impulse::DatasetSample> DatasetData;

    class Dataset {
    protected:
        Impulse::DatasetData data;
    public:
        ~Dataset();

        void addSample(Impulse::DatasetSample sample);

        void addSample(Impulse::DatasetSample *sample);

        Impulse::DatasetData *getSamples();

        Impulse::DatasetSample *getSampleAt(int index);

        void out(unsigned int limit = 10);

        unsigned int getSize();

        unsigned int getColumnsSize();
    };

}

#endif /* SRC_IMPULSE_DATASET_H_ */
