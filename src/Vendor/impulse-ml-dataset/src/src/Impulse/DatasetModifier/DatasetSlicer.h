#ifndef SRC_IMPULSE_DATASETSLICER_H_
#define SRC_IMPULSE_DATASETSLICER_H_

#include <vector>
#include <algorithm>
#include <cstdarg>
#include <string>
#include "../Dataset.h"

namespace Impulse {

    struct SlicedDataset {
        Impulse::Dataset input;
        Impulse::Dataset output;

        Eigen::MatrixXd &getInput() {
            if (this->inputVector.rows() == 0) {
                this->inputVector.resize(this->input.getSize(), this->input.getSampleAt(0)->getSize());
                for (unsigned int i = 0; i < this->input.getSize(); i++) {
                    this->inputVector.row(i) = this->input.getSampleAt(i)->exportToEigen();
                }
            }

            return this->inputVector;
        }

        Eigen::MatrixXd &getOutput() {
            if (this->outputVector.rows() == 0) {
                this->outputVector.resize(this->output.getSize(), this->output.getSampleAt(0)->getSize());
                for (unsigned int i = 0; i < this->output.getSize(); i++) {
                    this->outputVector.row(i) = this->output.getSampleAt(i)->exportToEigen();
                }
            }
            return this->outputVector;
        }

        Eigen::MatrixXd inputVector;
        Eigen::MatrixXd outputVector;
    };

    namespace DatasetModifier {

        class DatasetSlicer {
        protected:
            Impulse::Dataset *dataset;
            std::vector<int> inputColumns;
            std::vector<int> outputColumns;
        public:
            DatasetSlicer(Impulse::Dataset *_dataset);

            void addInputColumn(int columnIndex);

            void addOutputColumn(int columnIndex);

            Impulse::SlicedDataset slice();
        };

    }

}

#endif /* SRC_IMPULSE_DATASETSLICER_H_ */
