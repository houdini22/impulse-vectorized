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

#include "../../Vendor/impulse-ml-dataset/src/src/Impulse/DatasetModifier/DatasetSlicer.h"

#include "common.h"
#include "Math/common.h"
#include "Trainer/common.h"
#include "Layer/Abstract.h"
#include "Network.h"
#include "Builder.h"
#include "Serializer.h"
#include "Layer/Softmax.h"
#include "Layer/Relu.h"
#include "Layer/Logistic.h"
#include "Layer/Purelin.h"
#include "Math/Fmincg.h"
#include "Trainer/AbstractTrainer.h"
#include "Trainer/ConjugateGradientTrainer.h"

#endif //IMPULSE_NEURALNETWORK_INCLUDE_H
