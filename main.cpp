/*
#define EIGEN_USE_MKL_ALL
#define EIGEN_USE_BLAS
#define EIGEN_USE_THREADS
#define EIGEN_USE_GPU
#define EIGEN_USE_SYCL
*/

#include <iostream>
#include <cstdlib>
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

#include "src/Impulse/NeuralNetwork/include.h"

using namespace std::chrono;
using namespace Impulse::NeuralNetwork;

Impulse::SlicedDataset getDataset() {
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

    return dataset;
}

void test_logistic() {
    Impulse::SlicedDataset dataset = getDataset();

    Builder builder(400);
    builder.createLayer(100, Layer::TYPE_LOGISTIC);
    builder.createLayer(20, Layer::TYPE_LOGISTIC);
    builder.createLayer(10, Layer::TYPE_LOGISTIC);

    Network net = builder.getNetwork();

    //std::cout << "Forward:" << std::endl << net.forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Trainer::ConjugateGradientTrainer trainer(net);
    trainer.setLearningIterations(400);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.0);

    Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    trainer.train(dataset);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(t2 - t1).count();
    std::cout << "Time: " << duration << std::endl;

    std::cout << "Forward:" << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    Serializer serializer(net);
    serializer.toJSON("/home/hud/CLionProjects/impulse-vectorized/saved/logistic.json");
}

void test_softmax() {
    Impulse::SlicedDataset dataset = getDataset();

    Builder builder(400);
    builder.createLayer(100, Layer::TYPE_LOGISTIC);
    builder.createLayer(20, Layer::TYPE_LOGISTIC);
    builder.createLayer(10, Layer::TYPE_SOFTMAX);

    Network net = builder.getNetwork();

    Trainer::ConjugateGradientTrainer trainer(net);
    trainer.setLearningIterations(400);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.0);

    Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;

    std::cout << "Forward:" << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    trainer.train(dataset);
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<seconds>(t2 - t1).count();
    std::cout << "Time: " << duration << std::endl;

    std::cout << "Forward:" << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen()) << std::endl;

    Serializer serializer(net);
    serializer.toJSON("/home/hud/CLionProjects/impulse-vectorized/saved/softmax.json");
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

    Builder builder(2);
    builder.createLayer(3, Layer::TYPE_LOGISTIC);
    builder.createLayer(1, Layer::TYPE_LOGISTIC);

    Network net = builder.getNetwork();

    Impulse::DatasetSample sample({0, 1});
    Math::T_Matrix inputVector = sample.exportToEigen();
    std::cout << "Forward: " << net.forward(inputVector) << std::endl;

    Trainer::ConjugateGradientTrainer trainer(net);
    trainer.setLearningIterations(100);

    Trainer::CostGradientResult cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;

    trainer.train(slicedDataset);

    std::cout << "Forward: " << net.forward(inputVector) << std::endl;

    Impulse::DatasetSample sample2({1, 1});
    Math::T_Matrix inputVector2 = sample2.exportToEigen();
    std::cout << "Forward: " << net.forward(inputVector2) << std::endl;

    Serializer serializer(net);
    serializer.toJSON("/home/hud/CLionProjects/impulse-vectorized/saved/xor.json");
}

void test_logistic_load() {
    Builder builder = Builder::fromJSON("/home/hud/CLionProjects/impulse-vectorized/saved/logistic.json");
    Network net = builder.getNetwork();

    Impulse::SlicedDataset dataset = getDataset();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    std::cout << "Saved Forward: " << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen())
              << std::endl;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "Forward time: " << duration << std::endl;

    Trainer::ConjugateGradientTrainer trainer(net);

    Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;
}

void test_linear() {
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder(
            "/home/hud/CLionProjects/impulse-vectorized/data/linear.csv");
    Impulse::Dataset dataset = datasetBuilder.build();

    Impulse::DatasetModifier::DatasetSlicer slicer(&dataset);

    slicer.addInputColumn(0);
    slicer.addInputColumn(1);
    slicer.addOutputColumn(2);

    Impulse::SlicedDataset slicedDataset = slicer.slice();

    Builder builder(2);
    builder.createLayer(3, Layer::TYPE_PURELIN);
    builder.createLayer(1, Layer::TYPE_PURELIN);

    Network net = builder.getNetwork();

    Impulse::DatasetSample sample({2, 2});
    Math::T_Matrix inputVector = sample.exportToEigen();
    std::cout << "Forward: " << net.forward(inputVector) << std::endl;

    Trainer::ConjugateGradientTrainer trainer(net);
    trainer.setLearningIterations(100);

    Trainer::CostGradientResult cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;

    trainer.train(slicedDataset);

    std::cout << "Forward: " << net.forward(inputVector) << std::endl;

    Impulse::DatasetSample sample2({2, 2});
    Math::T_Matrix inputVector2 = sample2.exportToEigen();
    std::cout << "Forward: " << net.forward(inputVector2) << std::endl;

    Serializer serializer(net);
    serializer.toJSON("/home/hud/CLionProjects/impulse-vectorized/saved/linear.json");
}

void face() {
    // create dataset
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder1(
            "/media/hud/INTENSO/ML/facedb/exported/X.csv");
    Impulse::Dataset datasetInput = datasetBuilder1.build();

    Impulse::DatasetBuilder::CSVBuilder datasetBuilder2(
            "/media/hud/INTENSO/ML/facedb/exported/Y.csv");
    Impulse::Dataset datasetOutput = datasetBuilder2.build();

    Impulse::SlicedDataset dataset;
    dataset.input = datasetInput;
    dataset.output = datasetOutput;


}

int main() {
    //test_logistic();
    //test_softmax();
    test_linear();
    //test_logistic_load();
    //test_xor();
    //face();
    return 0;
}