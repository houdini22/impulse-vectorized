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
#include "src/Impulse/NeuralNetwork/NetworkSerializer.h"

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
    builder.createLayer(20, Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC);
    builder.createLayer(10, Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC);

    Impulse::NeuralNetwork::Network *net = builder.getNetwork();

    std::cout << "Forward:" << std::endl << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Impulse::NeuralNetwork::Trainer::AbstractTrainer trainer(net);
    trainer.setLearningIterations(2000);
    trainer.setLearningRate(0.001);
    trainer.setVerboseStep(50);

    double cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost << std::endl;

    trainer.train(dataset);

    std::cout << "Forward:" << std::endl << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;
}

void test_xor() {
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder(
            "/home/hud/CLionProjects/impulse-vectorized/data/xor.csv");
    Impulse::Dataset dataset = datasetBuilder.build();

    Impulse::DatasetModifier::DatasetSlicer slicer(&dataset);

    slicer.addInputColumn(0);
    slicer.addInputColumn(1);
    slicer.addOutputColumn(2);

    Impulse::SlicedDataset slicedDataset = slicer.slice();

    Impulse::NeuralNetwork::Builder::Builder builder(2);
    builder.createLayer(2, Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC);
    builder.createLayer(1, Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC);

    Impulse::NeuralNetwork::Network *net = builder.getNetwork();

    Impulse::DatasetSample sample({0, 1});
    Eigen::MatrixXd inputVector = sample.exportToEigen();
    std::cout << "Forward: " << net->forward(inputVector) << std::endl;

    Impulse::NeuralNetwork::Trainer::AbstractTrainer trainer(net);
    trainer.setLearningIterations(20000);
    trainer.setLearningRate(20);
    trainer.setVerboseStep(1000);

    double cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost << std::endl;

    trainer.train(slicedDataset);

    std::cout << "Forward: " << net->forward(inputVector) << std::endl;

    Impulse::DatasetSample sample2({1, 1});
    Eigen::MatrixXd inputVector2 = sample2.exportToEigen();
    std::cout << "Forward: " << net->forward(inputVector2) << std::endl;

    Impulse::NeuralNetwork::NetworkSerializer serializer(net);
    serializer.toJSON("/home/hud/CLionProjects/impulse-vectorized/saved/xor.json");
}

void test_xor_load() {
    Impulse::NeuralNetwork::Builder::Builder builder = Impulse::NeuralNetwork::Builder::Builder::fromJSON("/home/hud/CLionProjects/impulse-vectorized/saved/xor.json");
    Impulse::NeuralNetwork::Network * net = builder.getNetwork();

    Impulse::DatasetSample sample2({1, 1});
    Eigen::MatrixXd inputVector2 = sample2.exportToEigen();
    std::cout << "Saved Forward: " << net->forward(inputVector2) << std::endl;
}

int main() {
    //test_logistic();
    test_xor();
    test_xor_load();
    return 0;
}