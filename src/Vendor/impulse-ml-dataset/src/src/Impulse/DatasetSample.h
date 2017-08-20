#ifndef SRC_IMPULSE_DATASETSAMPLE_H_
#define SRC_IMPULSE_DATASETSAMPLE_H_

#include <string>
#include <vector>
#include <functional>
#include <iostream>

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace Impulse {

    class DatasetSample {
    protected:
        std::vector<std::string> rawData;
    public:
        DatasetSample(std::vector<std::string> vec);

        DatasetSample(std::vector<std::string> *vec);

        DatasetSample(std::vector<double> vec);

        DatasetSample(std::initializer_list<double> list);

        void out();

        std::string getColumnToString(int columnIndex);

        double getColumnToDouble(int columnIndex);

        void setColumn(int columnIndex, std::string value);

        void setColumn(int columnIndex, double value);

        void setColumn(int columnIndex,
                       std::function<std::string(std::string)> callback);

        void insertColumnAfter(int columnIndex, std::string value);

        void removeColumnAt(int columnIndex);

        unsigned int getSize();

        Impulse::DatasetSample copy();

        std::vector<double> exportToDoubleVector();

        Eigen::VectorXd exportToEigen();
    };

}

#endif /* SRC_IMPULSE_DATASETSAMPLE_H_ */
