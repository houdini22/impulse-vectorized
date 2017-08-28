/*#define EIGEN_USE_MKL_ALL
#define EIGEN_NO_DEBUG
#define EIGEN_VECTORIZE
#define VECTORIZE
#define EIGEN_VECTORIZE_FMA
#define EIGEN_VECTORIZE_SSE4_1
#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_AVX
#define EIGEN_VECTORIZE_AVX2
#define EIGEN_USE_BLAS
*/

#define EIGEN_USE_MKL_ALL
#define EIGEN_NO_DEBUG
#define EIGEN_VECTORIZE
#define VECTORIZE
#define EIGEN_VECTORIZE_FMA
#define EIGEN_VECTORIZE_SSE4_1
#define EIGEN_VECTORIZE_SSE4_2
#define EIGEN_VECTORIZE_AVX
#define EIGEN_VECTORIZE_AVX2
#define EIGEN_USE_BLAS

#include <iostream>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <chrono>
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

using namespace std::chrono;

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
    builder.createLayer(100, Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC);
    builder.createLayer(20, Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC);
    builder.createLayer(10, Impulse::NeuralNetwork::Layer::TYPE_LOGISTIC);

    Impulse::NeuralNetwork::Network *net = builder.getNetwork();

    std::cout << "Forward:" << std::endl << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Impulse::NeuralNetwork::Trainer::AbstractTrainer trainer(net);
    trainer.setLearningIterations(100);
    trainer.setLearningRate(0.001);
    trainer.setVerboseStep(50);

    double cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    trainer.train(dataset);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<seconds>( t2 - t1 ).count();
    std::cout << "Time: " << duration << std::endl;

    std::cout << "Forward:" << std::endl << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Impulse::NeuralNetwork::NetworkSerializer serializer(net);
    serializer.toJSON("/home/hud/CLionProjects/impulse-vectorized/saved/logistic.json");
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

void test_logistic_load() {
    Impulse::NeuralNetwork::Builder::Builder builder = Impulse::NeuralNetwork::Builder::Builder::fromJSON("/home/hud/CLionProjects/impulse-vectorized/saved/logistic.json");
    Impulse::NeuralNetwork::Network * net = builder.getNetwork();

    Impulse::DatasetBuilder::CSVBuilder datasetBuilder1(
            "/home/hud/CLionProjects/impulse-new/data/ex4data1_x.csv");
    Impulse::Dataset datasetInput = datasetBuilder1.build();
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder2(
            "/home/hud/CLionProjects/impulse-new/data/ex4data1_y.csv");
    Impulse::Dataset datasetOutput = datasetBuilder2.build();

    Impulse::SlicedDataset dataset;
    dataset.input = datasetInput;
    dataset.output = datasetOutput;

    std::cout << "Saved Forward: " << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Impulse::NeuralNetwork::Trainer::AbstractTrainer trainer(net);

    double cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost << std::endl;
}

int main() {
    test_logistic();
    //test_logistic_load();
    // test_xor();
    //test_xor_load();
    return 0;
}