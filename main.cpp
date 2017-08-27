//#define DEBUG 1

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
#include "src/Impulse/NeuralNetwork/Trainer/BatchGradientDescentTrainer.h"
#include "src/Impulse/NeuralNetwork/NetworkSerializer.h"
#include "src/Impulse/NeuralNetwork/Trainer/CojungateGradientTrainer.h"

void test_logistic_gradient_descent() {
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

    Impulse::NeuralNetwork::Trainer::ConjugateGradientTrainer trainer(net);
    trainer.setLearningIterations(500);
    trainer.setLearningRate(0.001);
    trainer.setVerboseStep(50);

    Impulse::NeuralNetwork::Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getError() << std::endl;

    trainer.train(dataset);

    std::cout << "Forward:" << std::endl << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Impulse::NeuralNetwork::NetworkSerializer serializer(net);
    serializer.toJSON("/home/hud/CLionProjects/impulse-vectorized/saved/logistic.json");
}

void test_logistic_conjugate_gradient() {
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

    /*Eigen::VectorXd theta = net->getRolledTheta();
    net->setRolledTheta(theta);

    std::cout << "Forward:" << std::endl << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    return;*/

    Impulse::NeuralNetwork::Trainer::ConjugateGradientTrainer trainer(net);
    trainer.setLearningIterations(500);
    trainer.setLearningRate(0.001);
    trainer.setVerboseStep(50);

    Impulse::NeuralNetwork::Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getError() << std::endl;

    trainer.train(dataset);

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

    Impulse::NeuralNetwork::Trainer::ConjugateGradientTrainer trainer(net);
    trainer.setLearningIterations(1000);
    trainer.setLearningRate(20);
    trainer.setVerboseStep(100);

    Impulse::NeuralNetwork::Trainer::CostGradientResult cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost.getError() << std::endl;

    clock_t begin = clock();
    trainer.train(slicedDataset);
    clock_t end = clock();

    std::cout << "Training time: " << (double(end - begin) / CLOCKS_PER_SEC) << std::endl;

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

    Impulse::NeuralNetwork::Trainer::ConjugateGradientTrainer trainer(net);

    Impulse::NeuralNetwork::Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getError() << std::endl;
}

int main() {
    //test_logistic_gradient_descent();
    //test_logistic_load();
    //test_xor();
    //test_xor_load();
    test_logistic_conjugate_gradient();
    return 0;
}