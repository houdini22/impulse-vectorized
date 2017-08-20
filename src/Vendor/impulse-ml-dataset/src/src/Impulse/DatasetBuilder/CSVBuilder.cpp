#include "CSVBuilder.h"

namespace Impulse {

    namespace DatasetBuilder {

        CSVBuilder::CSVBuilder(std::string path) {
            this->path = path;
        }

        CSVBuilder::~CSVBuilder() {
            this->closeFile();
        }

        void CSVBuilder::openFile() {
            this->fileHandle.open(this->path);
            if (!this->fileHandle.is_open()) {
                std::cout << "Cannot open file." << std::endl;
                throw "Cannot open file.";
            }
        }

        void CSVBuilder::closeFile() {
            if (this->fileHandle.is_open()) {
                this->fileHandle.close();
            }
        }

        std::vector<std::string> CSVBuilder::parseLine(std::string &line) {
            CSVState state = CSVState::UnquotedField;
            std::vector<std::string> fields{""};
            unsigned int i = 0; // index of the current field

            for (char c : line) {
                switch (state) {
                    case CSVState::UnquotedField:
                        switch (c) {
                            case ',': // end of field
                                fields.push_back("");
                                i++;
                                break;
                            case '"':
                                state = CSVState::QuotedField;
                                break;
                            default:
                                fields[i].push_back(c);
                                break;
                        }
                        break;
                    case CSVState::QuotedField:
                        switch (c) {
                            case '"':
                                state = CSVState::QuotedQuote;
                                break;
                            default:
                                fields[i].push_back(c);
                                break;
                        }
                        break;
                    case CSVState::QuotedQuote:
                        switch (c) {
                            case ',': // , after closing quote
                                fields.push_back("");
                                i++;
                                state = CSVState::UnquotedField;
                                break;
                            case '"': // "" -> "
                                fields[i].push_back('"');
                                state = CSVState::QuotedField;
                                break;
                            default:  // end of quote
                                state = CSVState::UnquotedField;
                                break;
                        }
                        break;
                }
            }

            return fields;
        }

        Impulse::Dataset CSVBuilder::build() {
            this->openFile();

            std::string line;
            while (std::getline(this->fileHandle, line)) {
                std::vector<std::string> fields = this->parseLine(line);
                this->dataset.addSample(this->createSample(fields));
            }

            this->closeFile();

            return this->dataset;
        }

    }

}
