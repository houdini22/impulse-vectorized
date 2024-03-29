#ifndef IMPULSE_NEURALNETWORK_INCLUDE_H
#define IMPULSE_NEURALNETWORK_INCLUDE_H

#include <eigen3/Eigen/Core>
#include <eigen3/Eigen/Dense>
#include "../../Vendor/json/src/json.hpp"
#include <string>
#include <vector>
#include <functional>
#include <algorithm>
#include <iostream>
#include <ctime>
#include <chrono>
#include <functional>
#include <cmath>
#include <fstream>
#include <utility>

#include "../../Vendor/impulse-ml-dataset/src/src/Impulse/Dataset/include.h"

#include "common.h"
#include "Math/common.h"
#include "utils.h"
#include "Trainer/common.h"
#include "Layer/BackPropagation/Abstract.h"
#include "Layer/BackPropagation/BackPropagation1DTo1D.h"
#include "Layer/BackPropagation/BackPropagationToMaxPool.h"
#include "Layer/BackPropagation/BackPropagationToConv.h"
#include "Layer/BackPropagation/BackPropagation3DTo1D.h"
#include "Layer/BackPropagation/Factory.h"
#include "Layer/Abstract.h"
#include "Layer/Abstract1D.h"
#include "Layer/Abstract3D.h"
#include "Network/Abstract.h"
#include "Network/ConvNetwork.h"
#include "Network/ClassifierNetwork.h"
#include "Builder/Abstract.h"
#include "Builder/ClassifierBuilder.h"
#include "Builder/ConvBuilder.h"
#include "Serializer.h"
#include "Layer/Softmax.h"
#include "Layer/Relu.h"
#include "Layer/Logistic.h"
#include "Layer/Purelin.h"
#include "Layer/Conv.h"
#include "Layer/MaxPool.h"
#include "Layer/FullyConnected.h"
#include "Math/Fmincg.h"
#include "Trainer/Abstract.h"
#include "Trainer/ConjugateGradient.h"
#include "Trainer/GradientDescent.h"
#include "Trainer/MiniBatchGradientDescent.h"

#endif //IMPULSE_NEURALNETWORK_INCLUDE_H
