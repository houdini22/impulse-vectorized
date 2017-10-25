#include "include.h"

namespace Impulse {

    namespace NeuralNetwork {

        Serializer::Serializer(Network::Abstract &net) : network(net) {}

        void Serializer::toJSON(T_String path) {
            nlohmann::json result;

            T_Dimension dim = this->network.getDimension();
            result["inputSize"] = {dim.width, dim.height, dim.depth};

            result["layers"] = {};
            for (T_Size i = 0; i < this->network.getSize(); i++) {
                result["layers"][i] = nlohmann::json::array(
                        {this->network.getLayer(i)->getSize(), this->network.getLayer(i)->getType()});
            }

            Math::T_Vector theta = this->network.getRolledTheta();
            result["weights"] = Math::vectorToRaw(theta);

            std::ofstream out(path);
            out << result.dump();
            out.close();
        }
    }
}
