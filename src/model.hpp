#ifndef MODEL_H
#define MODEL_H

#include <map>
#include <set>
#include <string>
#include <stdlib.h>

#include <json/json.hpp>

#include "graph.hpp"
#include "key.hpp"

using json = nlohmann::json;

class Model {
public:
    Model(void);
    Model(unsigned int feature, std::string feature_name, std::string type, std::string relation, std::string reference, std::map< bool, Model > const & submodels);
    Model(std::string feature_name, std::string type, std::string prediction, float loss, float complexity, Bitmask const & capture);

    std::set< Bitmask > partitions(void) const;
    size_t const hash(void) const;
    bool const operator==(Model const & other) const;

    std::string predict(Bitmask const & sample) const;

    std::string serialize(void) const;
    std::string serialize(int const spacing) const;
    json to_json(void) const;
private:
    Bitmask capture;

    bool test_integral(std::string const & string) const;
    bool test_rational(std::string const & string) const;

    // Non-terminal members
    unsigned int feature;
    std::string feature_name;
    std::string type;
    std::string relation;
    std::string reference;
    std::map< bool, Model > submodels;

    // Terminal members
    std::string prediction;
    float loss;
    float complexity;
};

namespace std {
  template <>
  struct hash< Model > {
    std::size_t operator()(Model const & model) const {
      return model.hash();
    }
  };
}

#endif