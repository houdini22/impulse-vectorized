#ifndef IMPULSE_VECTORIZED_ABSTRACT_H
#define IMPULSE_VECTORIZED_ABSTRACT_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>

namespace Impulse {

    namespace NeuralNetwork {

        namespace Layer {

            class Abstract {
            protected:
                unsigned int size;
                unsigned int prevSize = 0;
            public:
                Eigen::MatrixXd W;
                Eigen::VectorXd b;
                Eigen::MatrixXd A;
                Eigen::MatrixXd Z;
                Eigen::MatrixXd dW;
                Eigen::MatrixXd db;
                Eigen::MatrixXd vW;
                Eigen::MatrixXd vb;
                Eigen::MatrixXd sW;
                Eigen::MatrixXd sb;

                Abstract(unsigned int size, unsigned int prevSize) {
                    this->size = size;
                    this->prevSize = prevSize;

                    this->W.resize(this->size, this->prevSize);
                    this->W.setRandom();
                    this->W = this->W.array() * sqrt(2.0 / this->prevSize);

                    this->b.resize(this->size, 1);
                    this->b.setZero();

                    this->vW.resize(this->size, this->prevSize);
                    this->vW.setZero();

                    this->vb.resize(this->size, 1);
                    this->vb.setZero();

                    this->sW.resize(this->size, this->prevSize);
                    this->sW.setZero();

                    this->sb.resize(this->size, 1);
                    this->sb.setZero();
                }

                Eigen::MatrixXd forward(Eigen::MatrixXd input) {
                    this->Z = (this->W * input).colwise() + this->b;
                    this->A = this->activation(this->Z);
                    return this->A;
                }

                virtual Eigen::MatrixXd activation(Eigen::MatrixXd input) = 0;

                virtual Eigen::MatrixXd derivative() = 0;

                void updateParameters(double learningRate) {
                    double t = 2.0;
                    double epsilon = 1e-8;

                    this->vW = (0.9 * this->vW) + ((1.0 - 0.9) * this->dW);
                    this->vb = (0.9 * this->vb) + ((1.0 - 0.9) * this->db);

                    Eigen::MatrixXd vWcorrected = this->vW / (1.0 - (pow(0.9, t)));
                    Eigen::MatrixXd vbCorrected = this->vb / (1.0 - (pow(0.9, t)));

                    this->sW = (0.999 * this->sW) + ((1.0 - 0.999) * this->dW.unaryExpr([](const double x) {
                        return pow(x, 2.0);
                    }));
                    this->sb = (0.999 * this->sW) + ((1.0 - 0.999) * this->db.unaryExpr([](const double x) {
                        return pow(x, 2.0);
                    }));

                    Eigen::MatrixXd sWCorrected = this->sW / (1.0 - (pow(0.999, t)));
                    Eigen::MatrixXd sbCorrected = this->sb / (1.0 - pow(0.999, t));

                    this->W += learningRate * (vWcorrected.array() / (sWCorrected.unaryExpr([&epsilon](const double x) {
                        return sqrt(x + epsilon);
                    })).array()).array();
                    this->b += learningRate * (vbCorrected.array() / (sbCorrected.unaryExpr([&epsilon](const double x) {
                        return sqrt(x + epsilon);
                    })).array()).array();
                }

                unsigned int getSize() {
                    return this->size;
                }

                virtual std::string getType() = 0;
            };
        }

    }

}

#endif //IMPULSE_VECTORIZED_ABSTRACT_H
