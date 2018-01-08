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

#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"

#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetBuilder/CSVBuilder.h"
#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/Dataset.h"
#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

#include "src/Impulse/NeuralNetwork/include.h"
#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/Callback.h"
#include "src/Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/Modifier/MinMaxScaling.h"

using namespace std::chrono;
using namespace Impulse::NeuralNetwork;
using namespace cv;

Impulse::SlicedDataset getDataset() {
    // create dataset
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder1(
            "/home/hud/projekty/impulse-vectorized/data/ex4data1_x.csv");
    Impulse::Dataset datasetInput = datasetBuilder1.build();

    Impulse::DatasetBuilder::CSVBuilder datasetBuilder2(
            "/home/hud/projekty/impulse-vectorized/data/ex4data1_y.csv");
    Impulse::Dataset datasetOutput = datasetBuilder2.build();

    Impulse::SlicedDataset dataset;
    dataset.input = datasetInput;
    dataset.output = datasetOutput;

    return dataset;
}

/*void test_logistic() {
    Impulse::SlicedDataset dataset = getDataset();

    Builder builder(400);
    builder.createLayer(100, Layer::TYPE_LOGISTIC);
    builder.createLayer(20, Layer::TYPE_LOGISTIC);
    builder.createLayer(10, Layer::TYPE_LOGISTIC);

    Abstract net = builder.getNetwork();

    //std::cout << "Forward:" << std::endl << net.forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Trainer::ConjugateGradient trainer(net);
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
    serializer.toJSON("/home/hud/projekty/impulse-vectorized/saved/logistic.json");
}*/

void test_softmax_gradient_descent() {

    Impulse::SlicedDataset dataset = getDataset();

    Builder::ClassifierBuilder builder({400});

    builder.createLayer<Layer::Relu>([](auto *layer) {
        layer->setSize(100);
    });
    builder.createLayer<Layer::Relu>([](auto *layer) {
        layer->setSize(20);
    });
    builder.createLayer<Layer::Softmax>([](auto *layer) {
        layer->setSize(10);
    });

    Network::ClassifierNetwork net = builder.getNetwork();

    /*Builder::ClassifierBuilder builder = Builder::ClassifierBuilder::fromJSON("/home/hud/projekty/impulse-vectorized/saved/softmax.json");
    Network::ClassifierNetwork net = builder.getNetwork();*/

    Trainer::GradientDescent trainer(net);
    trainer.setLearningIterations(20000);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.0);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.05);

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
    serializer.toJSON("/home/hud/projekty/impulse-vectorized/saved/softmax-gradient-descent.json");
}

void test_softmax_cg() {

    Impulse::SlicedDataset dataset = getDataset();

    Builder::ClassifierBuilder builder({400});

    builder.createLayer<Layer::Logistic>([](auto *layer) {
        layer->setSize(300);
    });
    builder.createLayer<Layer::Softmax>([](auto *layer) {
        layer->setSize(10);
    });

    Network::ClassifierNetwork net = builder.getNetwork();

    /*Builder::ClassifierBuilder builder = Builder::ClassifierBuilder::fromJSON("/home/hud/projekty/impulse-vectorized/saved/softmax.json");
    Network::ClassifierNetwork net = builder.getNetwork();*/

    Trainer::ConjugateGradient trainer(net);
    trainer.setLearningIterations(300);
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
    serializer.toJSON("/home/hud/projekty/impulse-vectorized/saved/softmax.json");
}

void test_conv_backward() {
    Impulse::SlicedDataset dataset = getDataset();

    Builder::ConvBuilder builder({20, 20, 1});

    builder.createLayer<Layer::Conv>([](auto *layer) {
        layer->setFilterSize(3);
        layer->setPadding(0);
        layer->setStride(1);
        layer->setNumFilters(32);
    });

    /*builder.createLayer<Layer::Conv>([](auto *layer) {
        layer->setFilterSize(3);
        layer->setPadding(0);
        layer->setStride(1);
        layer->setNumFilters(32);
    });*/

    builder.createLayer<Layer::MaxPool>([](auto *layer) {
        layer->setFilterSize(2);
        layer->setStride(2);
    });

    builder.createLayer<Layer::Conv>([](auto *layer) {
        layer->setFilterSize(3);
        layer->setPadding(0);
        layer->setStride(1);
        layer->setNumFilters(64);
    });

    /*builder.createLayer<Layer::Conv>([](auto *layer) {
        layer->setFilterSize(3);
        layer->setPadding(0);
        layer->setStride(1);
        layer->setNumFilters(64);
    });*/

    builder.createLayer<Layer::MaxPool>([](auto *layer) {
        layer->setFilterSize(2);
        layer->setStride(2);
    });

    builder.createLayer<Layer::FullyConnected>([](auto *layer) {
        layer->setSize(512);
    });

    builder.createLayer<Layer::FullyConnected>([](auto *layer) {
        layer->setSize(256);
    });

    builder.createLayer<Layer::Softmax>([](auto *layer) {
        layer->setSize(10);
    });

    Network::ConvNetwork net = builder.getNetwork();

    Math::T_Matrix input = dataset.input.getSampleAt(0)->exportToEigen();
    Math::T_Matrix output = net.forward(input);
    std::cout << "OUTPUT: " << std::endl << output.rows() << "," << output.cols() << std::endl;

    Trainer::GradientDescent trainer(net);
    trainer.setLearningIterations(1);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.0);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.01);

    Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;

    trainer.train(dataset);
}

void test_conv_backward2() {
    Builder::ConvBuilder builder({2, 2, 1});

    builder.createLayer<Layer::Conv>([](auto *layer) {
        layer->setFilterSize(1);
        layer->setPadding(0);
        layer->setStride(1);
        layer->setNumFilters(4);
    });

    builder.createLayer<Layer::MaxPool>([](auto *layer) {
        layer->setFilterSize(2);
        layer->setStride(2);
    });

    builder.createLayer<Layer::FullyConnected>([](auto *layer) {
        layer->setSize(4);
    });

    Network::ConvNetwork net = builder.getNetwork();

    Impulse::DatasetBuilder::CSVBuilder datasetBuilder1(
            "/home/hud/projekty/impulse-vectorized/data/test1_x.csv");
    Impulse::Dataset datasetInput = datasetBuilder1.build();

    Impulse::DatasetBuilder::CSVBuilder datasetBuilder2(
            "/home/hud/projekty/impulse-vectorized/data/test1_y.csv");
    Impulse::Dataset datasetOutput = datasetBuilder2.build();

    Impulse::SlicedDataset dataset;
    dataset.input = datasetInput;
    dataset.output = datasetOutput;

    Math::T_Matrix netOutput = net.forward(datasetInput.getSampleAt(0)->exportToEigen());
    std::cout << "OUTPUT: " << std::endl << netOutput.rows() << "," << netOutput.cols() << std::endl;

    Trainer::GradientDescent trainer(net);
    trainer.setLearningIterations(1);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.0);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.01);

    trainer.train(dataset);
}

void test_conv() {
    //Impulse::SlicedDataset dataset = getDataset();

    Builder::ConvBuilder builder({5, 5, 3});

    builder.createLayer<Layer::Conv>([](auto *layer) {
        layer->setFilterSize(3);
        layer->setPadding(1);
        layer->setStride(2);
        layer->setNumFilters(2);
    });
    /*builder.createLayer<Layer::MaxPool>([](auto *layer) {
        layer->setFilterSize(2);
        layer->setStride(2);
    });*/

    /*builder.createLayer<Layer::FullyConnected>([](auto *layer) {
        layer->setSize(8);
    });
    builder.createLayer<Layer::FullyConnected>([](auto *layer) {
        layer->setSize(4);
    });
    */

    Network::ConvNetwork net = builder.getNetwork();
    // net.debug();

    Math::T_Matrix input;
    input.resize(5 * 5 * 3, 1);
    input.col(0) <<
                 2, 1, 1, 2, 1,
            2, 1, 1, 2, 0,
            1, 2, 0, 2, 0,
            2, 2, 1, 2, 2,
            2, 1, 2, 0, 0,
            1, 2, 2, 0, 2,
            0, 2, 2, 0, 1,
            2, 1, 0, 0, 0,
            0, 2, 1, 0, 0,
            0, 1, 1, 1, 1,
            2, 1, 2, 1, 2,
            0, 2, 0, 1, 0,
            2, 0, 0, 0, 0,
            2, 2, 1, 0, 0,
            0, 1, 2, 2, 0;


    std::cout << "INPUT: " << input << std::endl;
    Math::T_Matrix output = net.forward(input);

    std::cout << "OUTPUT: " << std::endl;
    std::cout << output << std::endl;

    Impulse::SlicedDataset dataset;

    Impulse::DatasetSample sampleInput({
                                               2, 1, 1, 2, 1,
                                               2, 1, 1, 2, 0,
                                               1, 2, 0, 2, 0,
                                               2, 2, 1, 2, 2,
                                               2, 1, 2, 0, 0,
                                               1, 2, 2, 0, 2,
                                               0, 2, 2, 0, 1,
                                               2, 1, 0, 0, 0,
                                               0, 2, 1, 0, 0,
                                               0, 1, 1, 1, 1,
                                               2, 1, 2, 1, 2,
                                               0, 2, 0, 1, 0,
                                               2, 0, 0, 0, 0,
                                               2, 2, 1, 0, 0,
                                               0, 1, 2, 2, 0
                                       });
    dataset.input.addSample(sampleInput);

    Impulse::DatasetSample outputSample({0, 1});
    dataset.output.addSample(outputSample);

    Trainer::GradientDescent trainer(net);
    trainer.setLearningIterations(1);
    trainer.setVerboseStep(1);
    trainer.setRegularization(0.0);
    trainer.setVerbose(true);
    trainer.setLearningRate(0.01);

    trainer.train(dataset);

    /*
    // MAX POOL TEST
    Math::T_Matrix test(16, 3);
    test.col(0) << 1, 1, 2, 4,
                    5, 6, 7, 8,
                    3, 2, 1, 0,
                    1, 2, 3, 4;

    test.col(1) << 9, 1, 12, 4,
            5, 6, 7, 8,
            13, 2, 14, 0,
            1, 2, 3, 4;

    test.col(2) << 11, 1, 12, 4,
            5, 6, 7, 8,
            13, 2, 1, 0,
            1, 2, 3, 14;

    Math::T_Matrix output = Utils::maxpool(test, 3, 4, 4, 2, 2, 2, 2);
    std::cout << output << std::endl;*/

    /*
    Trainer::ConjugateGradient trainer(net);
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
    serializer.toJSON("/home/hud/projekty/impulse-vectorized/saved/softmax.json");*/
}

/*void test_xor() {
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder(
            "/home/hud/projekty/impulse-vectorized/data/xor.csv");
    Impulse::Dataset dataset = datasetBuilder.build();

    Impulse::DatasetModifier::DatasetSlicer slicer(&dataset);

    slicer.addInputColumn(0);
    slicer.addInputColumn(1);
    slicer.addOutputColumn(2);

    Impulse::SlicedDataset slicedDataset = slicer.slice();

    Builder builder(2);
    builder.createLayer(3, Layer::TYPE_LOGISTIC);
    builder.createLayer(1, Layer::TYPE_LOGISTIC);

    Abstract net = builder.getNetwork();

    Impulse::DatasetSample sample({0, 1});
    Math::T_Matrix inputVector = sample.exportToEigen();
    std::cout << "Forward: " << net.forward(inputVector) << std::endl;

    Trainer::ConjugateGradient trainer(net);
    trainer.setLearningIterations(100);

    Trainer::CostGradientResult cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;

    trainer.train(slicedDataset);

    std::cout << "Forward: " << net.forward(inputVector) << std::endl;

    Impulse::DatasetSample sample2({1, 1});
    Math::T_Matrix inputVector2 = sample2.exportToEigen();
    std::cout << "Forward: " << net.forward(inputVector2) << std::endl;

    Serializer serializer(net);
    serializer.toJSON("/home/hud/projekty/impulse-vectorized/saved/xor.json");
}

void test_logistic_load() {
    Builder builder = Builder::fromJSON("/home/hud/projekty/impulse-vectorized/saved/logistic.json");
    Abstract net = builder.getNetwork();

    Impulse::SlicedDataset dataset = getDataset();

    high_resolution_clock::time_point t1 = high_resolution_clock::now();
    std::cout << "Saved Forward: " << std::endl << net.forward(dataset.input.getSampleAt(0)->exportToEigen())
              << std::endl;
    high_resolution_clock::time_point t2 = high_resolution_clock::now();
    auto duration = duration_cast<microseconds>(t2 - t1).count();
    std::cout << "Forward time: " << duration << std::endl;

    Trainer::ConjugateGradient trainer(net);

    Trainer::CostGradientResult cost = trainer.cost(dataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;
}

void test_linear() {
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder(
            "/home/hud/projekty/impulse-vectorized/data/linear.csv");
    Impulse::Dataset dataset = datasetBuilder.build();

    Impulse::DatasetModifier::DatasetSlicer slicer(&dataset);

    slicer.addInputColumn(0);
    slicer.addInputColumn(1);
    slicer.addOutputColumn(2);

    Impulse::SlicedDataset slicedDataset = slicer.slice();

    Builder builder(2);
    builder.createLayer(3, Layer::TYPE_PURELIN);
    builder.createLayer(1, Layer::TYPE_PURELIN);

    Abstract net = builder.getNetwork();

    Impulse::DatasetSample sample({2, 2});
    Math::T_Matrix inputVector = sample.exportToEigen();
    std::cout << "Forward: " << net.forward(inputVector) << std::endl;

    Trainer::ConjugateGradient trainer(net);
    trainer.setLearningIterations(100);

    Trainer::CostGradientResult cost = trainer.cost(slicedDataset);
    std::cout << "Cost: " << cost.getCost() << std::endl;

    trainer.train(slicedDataset);

    std::cout << "Forward: " << net.forward(inputVector) << std::endl;

    Impulse::DatasetSample sample2({2, 2});
    Math::T_Matrix inputVector2 = sample2.exportToEigen();
    std::cout << "Forward: " << net.forward(inputVector2) << std::endl;

    Serializer serializer(net);
    serializer.toJSON("/home/hud/projekty/impulse-vectorized/saved/linear.json");
}

void face() {
    // create dataset
    Impulse::DatasetBuilder::CSVBuilder datasetBuilder1(
            "/media/hud/INTENSO/ML/facedb/exported_all_scale_0_5/X.csv");
    Impulse::Dataset datasetInput = datasetBuilder1.build();

    std::cout << "X loaded." << std::endl;

    Impulse::DatasetModifier::Modifier::MinMaxScaling mod(&datasetInput);
    mod.apply();

    std::cout << "X modified." << std::endl;

    Impulse::DatasetBuilder::CSVBuilder datasetBuilder2(
            "/media/hud/INTENSO/ML/facedb/exported_all_scale_0_5/Y.csv");
    Impulse::Dataset datasetOutput = datasetBuilder2.build();

    std::cout << "Y loaded." << std::endl;

    Impulse::SlicedDataset dataset;
    dataset.input = datasetInput;
    dataset.output = datasetOutput;

    Builder builder(dataset.input.getColumnsSize());
    builder.createLayer(300, Layer::TYPE_LOGISTIC);
    builder.createLayer(300, Layer::TYPE_LOGISTIC);
    builder.createLayer(300, Layer::TYPE_LOGISTIC);
    builder.createLayer(300, Layer::TYPE_LOGISTIC);
    builder.createLayer(300, Layer::TYPE_LOGISTIC);
    builder.createLayer(300, Layer::TYPE_LOGISTIC);
    builder.createLayer(4, Layer::TYPE_PURELIN);

    Abstract net = builder.getNetwork();

    //std::cout << "Forward:" << std::endl << net.forward(datasetInput.getSampleAt(0)->exportToEigen()) << std::endl;

    Trainer::ConjugateGradient trainer(net);
    trainer.setLearningIterations(200);
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
    serializer.toJSON("/home/hud/projekty/impulse-vectorized/saved/face3_2.json");
}

void videoFace() {

    Builder builder = Builder::fromJSON("/home/hud/projekty/impulse-vectorized/saved/face3_2.json");
    Abstract net = builder.getNetwork();

    VideoCapture cap(0); // capture from default camera
    Mat frame;

    namedWindow("Face view",
                CV_WINDOW_AUTOSIZE |
                CV_WINDOW_FREERATIO |
                CV_GUI_EXPANDED);

    // Loop to capture frames
    while (cap.read(frame)) {
        Mat resized, grayed;
        resize(frame, resized, Size(384 / 2, 286 / 2));
        cvtColor(resized, grayed, CV_RGB2GRAY);
        std::vector<double> vec;

        for (int y = 0; y < grayed.rows; y++) {
            for (int x = 0; x < grayed.cols; x++) {
                auto pixelValue = (double) grayed.at<uchar>(y, x);
                vec.push_back(pixelValue);
            }
        }

        double min = 256.0;
        double max = -1.0;

        for (unsigned long i = 0; i < vec.size(); i++) {
            double value = vec.at(i);
            if (value < min) {
                min = value;
            } else if (value > max) {
                max = value;
            }
        }

        Math::T_Matrix input(vec.size(), 1);

        for (unsigned long i = 0; i < vec.size(); i++) {
            double value = vec.at(i);
            double newValue = (value - min) / (max - min);
            input(i, 0) = newValue;
        }

        Math::T_Matrix prediction = net.forward(input);

        circle(grayed, Point((int) prediction(0, 0), (int) prediction(1, 0)), 5, Scalar(255, 0, 0));
        circle(grayed, Point((int) prediction(2, 0), (int) prediction(3, 0)), 5, Scalar(255, 0, 0));

        imshow("Face view", grayed);

        if (waitKey(30) >= 0) // spacebar
            break;
    }
}*/

void test_test() {

    Builder::ConvBuilder builder({32, 32, 16});

    builder.createLayer<Layer::MaxPool>([](auto *layer) {
        layer->setFilterSize(2);
        layer->setStride(2);
    });

    Network::ConvNetwork net = builder.getNetwork();
    net.debug();
}

int main() {
    //test_logistic();
    //test_softmax_gradient_descent();
    //test_softmax_cg();
    //test_conv();
    //test_linear();
    //test_logistic_load();
    //test_xor();
    //face();
    //videoFace();
    //test_test();
    //test_conv_backward();
    test_conv_backward2();
    return 0;
}