#ifndef NETWORKSERIALIZER2_H
#define NETWORKSERIALIZER2_H

#include <fstream>
#include <string>
#include <iostream>

#include "Network.h"
#include "../../Vendor/json.hpp"

using json = nlohmann::json;

namespace Impulse {

    namespace NeuralNetwork {

        class NetworkSerializer {
        protected:
            Impulse::NeuralNetwork::Network *network;
        public:

            NetworkSerializer(Impulse::NeuralNetwork::Network *net) {
                this->network = net;
            }

            void toJSON(std::string path) {
                /*json result;

                result["inputSize"] = this->network->getInputSize();
                result["layers"] = {};

                for(unsigned int i = 0; i < this->network->getSize(); i++) {
                    result["layers"][i] = json::array({this->network->getLayer(i)->getSize(), this->network->getLayer(i)->getType()});
                }

                result["weights"] = this->network->getRolledTheta();

                std::ofstream out(path);
                out << result.dump();
                out.close();*/
            }
        };

    }

}

#endif
