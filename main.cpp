//#define DEBUG 1

//#define EIGEN_DONT_VECTORIZE
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

void test_simple() {
    Impulse::NeuralNetwork::Builder::Builder builder(3);
    builder.createLayer(2, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);
    builder.createLayer(2, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);

    Impulse::NeuralNetwork::Network *net = builder.getNetwork();

    Impulse::DatasetSample sample({0, 0.5, 1});
    Eigen::MatrixXd inputVector = sample.exportToEigen();
    std::cout << net->forward(inputVector) << std::endl;
}

void test_logistic() {
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

    Impulse::NeuralNetwork::Builder::Builder builder(400);
    builder.createLayer(20, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);
    builder.createLayer(10, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);

    Impulse::NeuralNetwork::Network *net = builder.getNetwork();

    std::cout << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;
}

void test_xor() {
    /*Impulse::DatasetBuilder::CSVBuilder datasetBuilder(
            "/home/hud/CLionProjects/impulse-vectorized/data/test.csv");
    */
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder(
            "/home/hud/CLionProjects/impulse-vectorized/data/xor.csv");
    Impulse::Dataset dataset = datasetBuilder.build();

    Impulse::DatasetModifier::DatasetSlicer slicer(&dataset);

    slicer.addInputColumn(0);
    slicer.addInputColumn(1);
    slicer.addOutputColumn(2);

    /*slicer.addInputColumn(0);
    slicer.addInputColumn(1);
    slicer.addInputColumn(2);
    slicer.addInputColumn(3);
    slicer.addInputColumn(4);
    slicer.addOutputColumn(5);
    slicer.addOutputColumn(6);
    slicer.addOutputColumn(7);*/

    Impulse::SlicedDataset slicedDataset = slicer.slice();

    Impulse::NeuralNetwork::Builder::Builder builder(2);
    builder.createLayer(2, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);
    builder.createLayer(1, Impulse::NeuralNetwork::Layer::TYPE_SIGMOID);

    Impulse::NeuralNetwork::Network *net = builder.getNetwork();

    Impulse::DatasetSample sample({0, 1});
    Eigen::MatrixXd inputVector = sample.exportToEigen();
    std::cout << "Forward: " << net->forward(inputVector) << std::endl;

    Impulse::NeuralNetwork::Trainer::AbstractTrainer trainer(net);
    trainer.setLearningIterations(100);
    trainer.setLearningRate(0.1);

    double cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost << std::endl;

    trainer.train(slicedDataset);

    std::cout << "Forward: " << net->forward(inputVector) << std::endl;
}

int main() {
    //test_simple();
    //test_logistic();
    test_xor();

    return 0;
}