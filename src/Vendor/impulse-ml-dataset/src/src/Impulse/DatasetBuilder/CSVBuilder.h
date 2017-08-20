#ifndef SRC_IMPULSE_DATASET_BUILDER_CSV_H_
#define SRC_IMPULSE_DATASET_BUILDER_CSV_H_

#include <fstream>
#include <string>
#include <fstream>
#include <iostream>
#include "BaseBuilder.h"
#include "../Dataset.h"

namespace Impulse {

    namespace DatasetBuilder {

        enum class CSVState {
            UnquotedField, QuotedField, QuotedQuote
        };

        class CSVBuilder : public Impulse::DatasetBuilder::BaseBuilder {

        protected:
            std::string path;
            std::ifstream fileHandle;

            void openFile();

            void closeFile();

            std::vector<std::string> parseLine(std::string &line);

        public:
            CSVBuilder(std::string p);

            virtual ~CSVBuilder();

            virtual Impulse::Dataset build();
        };

    }

}

#endif /* SRC_IMPULSE_DATASET_BUILDER_CSV_H_ */
