#include "DatasetSample.h"

namespace Impulse {

    DatasetSample::DatasetSample(std::vector<std::string> vec) {
        this->rawData = vec;
    }

    DatasetSample::DatasetSample(std::vector<std::string> *vec) {
        this->rawData = std::vector<std::string>(vec->begin(), vec->end());
    }

    DatasetSample::DatasetSample(std::vector<double> vec) {
        for (unsigned int i = 0; i < vec.size(); i++) {
            this->rawData.push_back(std::to_string(vec.at(i)));
        }
    }

    DatasetSample::DatasetSample(std::initializer_list<double> list) {
        for (auto i = list.begin(); i != list.end(); i++) {
            this->rawData.push_back(std::to_string(*i));
        }
    }

    void DatasetSample::out() {
        for (unsigned int i = 0; i < this->rawData.size() - 1; i++) {
            std::cout << this->rawData.at(i) << ',';
        }
        std::cout << this->rawData.at(this->rawData.size() - 1) << std::endl;
    }

    std::string DatasetSample::getColumnToString(int columnIndex) {
        return this->rawData.at(columnIndex);
    }

    double DatasetSample::getColumnToDouble(int columnIndex) {
        return std::stod(this->getColumnToString(columnIndex));
    }

    void DatasetSample::setColumn(int columnIndex, std::string value) {
        this->rawData.at(columnIndex) = value;
    }

    void DatasetSample::setColumn(int columnIndex, double value) {
        this->rawData.at(columnIndex) = std::to_string(value);
    }

    void DatasetSample::setColumn(int columnIndex,
                                  std::function<std::string(std::string)> callback) {
        this->rawData.at(columnIndex) = callback(this->rawData.at(columnIndex));
    }

    void DatasetSample::insertColumnAfter(int columnIndex, std::string value) {
        this->rawData.insert(this->rawData.begin() + columnIndex + 1, value);
    }

    void DatasetSample::removeColumnAt(int columnIndex) {
        this->rawData.erase(this->rawData.begin() + columnIndex);
    }

    unsigned int DatasetSample::getSize() {
        return this->rawData.size();
    }

    Impulse::DatasetSample DatasetSample::copy() {
        std::vector<std::string> copiedData(this->rawData);
        Impulse::DatasetSample result(copiedData);
        return result;
    }

    std::vector<double> DatasetSample::exportToDoubleVector() {
        std::vector<double> result;
        for (unsigned int i = 0; i < this->getSize(); i++) {
            result.push_back(this->getColumnToDouble(i));
        }
        return result;
    }

    Eigen::VectorXd DatasetSample::exportToEigen() {
        Eigen::VectorXd result(this->getSize());
        for (unsigned int i = 0; i < this->getSize(); i++) {
            result(i) = this->getColumnToDouble(i);
        }
        return result;
    }
}
