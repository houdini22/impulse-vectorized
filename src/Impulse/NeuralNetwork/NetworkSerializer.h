#ifndef NETWORK_SERIALIZER_H
#define NETWORK_SERIALIZER_H

#include <fstream>
#include <string>
#include <iostream>

#include "Network.h"
#include "../../Vendor/json.hpp"

using json = nlohmann::json;

using Impulse::NeuralNetwork::Math::T_Vector;

namespace Impulse {

    namespace NeuralNetwork {

        class NetworkSerializer {
        protected:
            Impulse::NeuralNetwork::Network *network;
        public:

            NetworkSerializer(Impulse::NeuralNetwork::Network *net) {
                this->network = net;
            }

            void toJSON(T_String path) {
                json result;

                result["inputSize"] = this->network->getInputSize();
                result["layers"] = {};

                for(unsigned int i = 0; i < this->network->getSize(); i++) {
                    result["layers"][i] = json::array({this->network->getLayer(i)->getSize(), this->network->getLayer(i)->getType()});
                }

                T_Vector weights = this->network->getRolledTheta();

                result["weights"] = std::vector<double>(weights.data(), weights.data() + weights.rows() * weights.cols());

                std::ofstream out(path);
                out << result.dump();
                out.close();
            }
        };
    }
}

#endif
