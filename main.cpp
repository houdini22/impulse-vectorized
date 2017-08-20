#include <iostream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <ios>
#include <ctime>
#include <experimental/filesystem>
#include <boost/filesystem.hpp>
#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/CSVBuilder.h"
#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset.h"
#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

#include "src/Impulse/NeuralNetwork/Builder/Builder.h"

int main() {
    // create dataset
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder1(
            "/home/hud/CLionProjects/impulse-new/data/ex4data1_x.csv");
    Impulse::Dataset datasetInput = datasetBuilder1.build();

    Impulse::DatasetBuilder::CSVBuilder datasetBuilder2(
            "/home/hud/CLionProjects/impulse-new/data/ex4data1_y.csv");
    Impulse::Dataset datasetOutput = datasetBuilder2.build();

    Impulse::SlicedDataset dataset;
    dataset.input = datasetInput;
    dataset.output = datasetOutput;

    Impulse::NeuralNetwork::Builder::Builder builder;
    builder.createLayer(400, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);
    builder.createLayer(20, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);
    builder.createLayer(10, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);

    Impulse::NeuralNetwork::Network * net = builder.getNetwork();
    //net->forward(datasetInput.getSampleAt(0)->exportToEigen());
    std::cout << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    return 0;
}