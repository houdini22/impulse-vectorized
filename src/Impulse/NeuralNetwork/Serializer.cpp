#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        Serializer::Serializer(Impulse::NeuralNetwork::Network *net) {
            this->network = net;
        }

        void Serializer::toJSON(T_String path) {
            nlohmann::json result;

            result["inputSize"] = this->network->getInputSize();

            result["layers"] = {};
            for (T_Size i = 0; i < this->network->getSize(); i++) {
                result["layers"][i] = nlohmann::json::array(
                        {this->network->getLayer(i)->getSize(), this->network->getLayer(i)->getType()});
            }

            Math::T_Vector theta = this->network->getRolledTheta();
            result["weights"] = Math::vectorToRaw(theta);

            std::ofstream out(path);
            out << result.dump();
            out.close();
        }
    }
}
