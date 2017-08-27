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

/*
 *  Impulse::NeuralNetwork::Trainer::BatchGradientDescent trainer(net);
    trainer.setLearningIterations(10000);
    trainer.setLearningRate(0.001);
    trainer.setVerboseStep(100);

    OUTPUT:

orward:
0.604009
0.638069
0.492909
0.431826
0.586219
0.417775
0.520309
0.530336
0.454752
0.544087
Cost: 7.39909
Starting training with 10000 iterations.
Iteration: 100 | Error:3.2185
Iteration: 200 | Error:2.75358
Iteration: 300 | Error:1.75667
Iteration: 400 | Error:1.2364
Iteration: 500 | Error:0.935432
Iteration: 600 | Error:0.758254
Iteration: 700 | Error:0.654659
Iteration: 800 | Error:0.586238
Iteration: 900 | Error:0.536123
Iteration: 1000 | Error:0.496824
Iteration: 1100 | Error:0.464506
Iteration: 1200 | Error:0.436998
Iteration: 1300 | Error:0.412981
Iteration: 1400 | Error:0.391675
Iteration: 1500 | Error:0.372559
Iteration: 1600 | Error:0.355199
Iteration: 1700 | Error:0.339537
Iteration: 1800 | Error:0.325323
Iteration: 1900 | Error:0.312296
Iteration: 2000 | Error:0.300171
Iteration: 2100 | Error:0.288824
Iteration: 2200 | Error:0.278401
Iteration: 2300 | Error:0.268983
Iteration: 2400 | Error:0.260491
Iteration: 2500 | Error:0.252778
Iteration: 2600 | Error:0.245681
Iteration: 2700 | Error:0.23904
Iteration: 2800 | Error:0.232773
Iteration: 2900 | Error:0.226866
Iteration: 3000 | Error:0.22129
Iteration: 3100 | Error:0.215983
Iteration: 3200 | Error:0.210851
Iteration: 3300 | Error:0.205996
Iteration: 3400 | Error:0.201385
Iteration: 3500 | Error:0.196951
Iteration: 3600 | Error:0.19269
Iteration: 3700 | Error:0.188439
Iteration: 3800 | Error:0.184245
Iteration: 3900 | Error:0.17996
Iteration: 4000 | Error:0.17608
Iteration: 4100 | Error:0.172564
Iteration: 4200 | Error:0.169266
Iteration: 4300 | Error:0.166127
Iteration: 4400 | Error:0.163167
Iteration: 4500 | Error:0.160357
Iteration: 4600 | Error:0.157626
Iteration: 4700 | Error:0.15499
Iteration: 4800 | Error:0.152467
Iteration: 4900 | Error:0.15003
Iteration: 5000 | Error:0.147798
Iteration: 5100 | Error:0.145708
Iteration: 5200 | Error:0.143739
Iteration: 5300 | Error:0.141863
Iteration: 5400 | Error:0.140029
Iteration: 5500 | Error:0.138224
Iteration: 5600 | Error:0.136442
Iteration: 5700 | Error:0.134683
Iteration: 5800 | Error:0.132943
Iteration: 5900 | Error:0.131262
Iteration: 6000 | Error:0.129688
Iteration: 6100 | Error:0.12832
Iteration: 6200 | Error:0.127012
Iteration: 6300 | Error:0.125783
Iteration: 6400 | Error:0.12464
Iteration: 6500 | Error:0.123572
Iteration: 6600 | Error:0.122546
Iteration: 6700 | Error:0.121556
Iteration: 6800 | Error:0.120601
Iteration: 6900 | Error:0.119676
Iteration: 7000 | Error:0.118779
Iteration: 7100 | Error:0.117899
Iteration: 7200 | Error:0.117013
Iteration: 7300 | Error:0.116124
Iteration: 7400 | Error:0.11524
Iteration: 7500 | Error:0.114397
Iteration: 7600 | Error:0.113747
Iteration: 7700 | Error:0.113045
Iteration: 7800 | Error:0.112238
Iteration: 7900 | Error:0.111457
Iteration: 8000 | Error:0.110723
Iteration: 8100 | Error:0.110025
Iteration: 8200 | Error:0.109337
Iteration: 8300 | Error:0.108644
Iteration: 8400 | Error:0.107954
Iteration: 8500 | Error:0.107191
Iteration: 8600 | Error:0.106453
Iteration: 8700 | Error:0.105674
Iteration: 8800 | Error:0.104814
Iteration: 8900 | Error:0.104239
Iteration: 9000 | Error:0.103711
Iteration: 9100 | Error:0.103184
Iteration: 9200 | Error:0.102659
Iteration: 9300 | Error:0.102143
Iteration: 9400 | Error:0.101659
Iteration: 9500 | Error:0.101193
Iteration: 9600 | Error:0.100729
Iteration: 9700 | Error:0.100275
Iteration: 9800 | Error:0.0998317
Iteration: 9900 | Error:0.0993932
Iteration: 10000 | Error:0.0989532
Training ended after 10000 iterations with error = 0.0989532.
Training time: 7614.02
Forward:
1.22989e-08
 0.00316424
0.000284675
 2.9568e-09
0.000222132
 0.00114751
0.000468642
3.21853e-05
5.83968e-05
    0.99808

 */

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

    Impulse::NeuralNetwork::Trainer::BatchGradientDescent trainer(net);
    trainer.setLearningIterations(10000);
    trainer.setLearningRate(0.001);
    trainer.setVerboseStep(100);

    Impulse::NeuralNetwork::Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getError() << std::endl;

    clock_t begin = clock();
    trainer.train(dataset);
    clock_t end = clock();

    std::cout << "Training time: " << (double(end - begin) / CLOCKS_PER_SEC) << std::endl;

    std::cout << "Forward:" << std::endl << net->forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Impulse::NeuralNetwork::NetworkSerializer serializer(net);
    serializer.toJSON("/home/hud/CLionProjects/impulse-vectorized/saved/logistic.json");
}

/*
 * void test_logistic_conjugate_gradient() {
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

*/
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
    trainer.setLearningIterations(5000);
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
    test_logistic_gradient_descent();
    //test_logistic_load();
    //test_xor();
    //test_xor_load();
    //test_logistic_conjugate_gradient();
    return 0;
}