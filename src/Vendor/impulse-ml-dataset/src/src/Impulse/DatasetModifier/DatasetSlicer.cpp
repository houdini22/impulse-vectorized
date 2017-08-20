#include "DatasetSlicer.h"

namespace Impulse {

    namespace DatasetModifier {

        DatasetSlicer::DatasetSlicer(Impulse::Dataset *_dataset) :
                dataset(_dataset) {

        }

        Impulse::SlicedDataset DatasetSlicer::slice() {
            Impulse::Dataset input;
            Impulse::Dataset output;

            Impulse::DatasetData *samples = this->dataset->getSamples();

            for (unsigned int i = 0; i < samples->size(); i++) {
                Impulse::DatasetSample oldSample = samples->at(i);
                std::vector<std::string> newInputSampleData;
                std::vector<std::string> newOutputSampleData;

                unsigned int j;

                for (j = 0; j < this->inputColumns.size(); j++) {
                    newInputSampleData.push_back(
                            oldSample.getColumnToString(this->inputColumns.at(j)));
                }
                Impulse::DatasetSample newInputSample(newInputSampleData);
                input.addSample(newInputSample);

                for (j = 0; j < this->outputColumns.size(); j++) {
                    newOutputSampleData.push_back(
                            oldSample.getColumnToString(this->outputColumns.at(j)));
                }

                Impulse::DatasetSample newOutputSample(newOutputSampleData);
                output.addSample(newOutputSample);
            }

            Impulse::SlicedDataset result;
            result.input = input;
            result.output = output;

            return result;
        }

        void DatasetSlicer::addInputColumn(int columnIndex) {
            this->inputColumns.push_back(columnIndex);
        }

        void DatasetSlicer::addOutputColumn(int columnIndex) {
            this->outputColumns.push_back(columnIndex);
        }

    }

}
