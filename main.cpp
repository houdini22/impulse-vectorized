//#define EIGEN_DONT_VECTORIZE

// SSE>2 doesn't affect these tests
#ifndef EIGEN_DONT_VECTORIZE // Not needed with Intel C++ Compiler XE 15.0
#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_SSE4_1
#define EIGEN_VECTORIZE_SSSE3
#define EIGEN_VECTORIZE_SSE3
#endif

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

#include "src/Impulse/NeuralNetwork/Trainer/AbstractTrainer.h"

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
    builder.createLayer(3, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);
    builder.createLayer(2, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);
    builder.createLayer(1, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);

    Impulse::NeuralNetwork::Network * net = builder.getNetwork();

    Impulse::DatasetSample sample({0, 0.5, 1});
    Eigen::MatrixXd inputVector = sample.exportToEigen();
    std::cout << net->forward(inputVector) << std::endl;

    /*Impulse::NeuralNetwork::Network * net = builder.getNetwork();
    //net->forward(datasetInput.getSampleAt(0)->exportToEigen());
    std::cout << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Impulse::NeuralNetwork::Trainer::AbstractTrainer trainer(net);
    Impulse::NeuralNetwork::Trainer::CostGradientResult result = trainer.cost(dataset);
    std::cout << result.getCost() << std::endl;

    trainer.train(dataset);*/

    return 0;
}