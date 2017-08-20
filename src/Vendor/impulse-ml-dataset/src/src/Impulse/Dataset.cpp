#include "Dataset.h"

namespace Impulse {

    Dataset::~Dataset() {
        this->data.clear();
    }

    void Dataset::addSample(Impulse::DatasetSample sample) {
        this->data.push_back(sample);
    }

    void Dataset::addSample(Impulse::DatasetSample *sample) {
        this->data.push_back(sample->copy());
    }

    void Dataset::out(unsigned int limit) {
        for (unsigned int i = 0; i < this->data.size(); i++) {
            if (limit > 0)
                if (i + 1 == limit)
                    break;

            this->data.at(i).out();
        }
        std::cout << "---" << std::endl;
    }

    DatasetData *Dataset::getSamples() {
        return &this->data;
    }

    Impulse::DatasetSample *Dataset::getSampleAt(int index) {
        return &this->data.at(index);
    }

    unsigned int Dataset::getSize() {
        return this->data.size();
    }

    unsigned int Dataset::getColumnsSize() {
        if (this->getSize() > 0) {
            return this->data.at(0).getSize();
        }
        return 0;
    }

}
