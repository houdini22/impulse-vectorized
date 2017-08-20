#include "DatasetSplitter.h"

namespace Impulse {

    namespace DatasetModifier {

        DatasetSplitter::DatasetSplitter(Impulse::SlicedDataset *_dataset) {
            this->slicedDataset = _dataset;
        }

        Impulse::SplittedDataset DatasetSplitter::split(double ratio) {
            Impulse::SplittedDataset result;

            for (unsigned int i = 0; i < this->slicedDataset->input.getSize(); i++) {
                double stepRatio = (i + 1)
                                   / (double) this->slicedDataset->input.getSize();
                if (ratio >= stepRatio) {
                    result.primary.input.addSample(
                            this->slicedDataset->input.getSampleAt(i));
                    result.primary.output.addSample(
                            this->slicedDataset->output.getSampleAt(i));
                } else {
                    result.secondary.input.addSample(
                            this->slicedDataset->input.getSampleAt(i));
                    result.secondary.output.addSample(
                            this->slicedDataset->output.getSampleAt(i));
                }
            }

            return result;
        }

    }

}
