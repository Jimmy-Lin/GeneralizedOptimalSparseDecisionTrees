#include "model.hpp"

Model::Model(void) {}

Model::Model(unsigned int feature, std::string feature_name, std::string type, std::string relation, std::string reference, std::map< bool, Model > const & submodels) :
    feature(feature), feature_name(feature_name), type(type), relation(relation), reference(reference), submodels(submodels) {}

Model::Model(std::string feature_name, std::string type, std::string prediction, float loss, float complexity, Bitmask const & capture) :
    feature_name(feature_name), type(type), prediction(prediction), loss(loss), complexity(complexity), capture(capture) {}


bool Model::test_rational(std::string const & string) const {
    std::string::const_iterator it = string.begin();
    bool decimalPoint = false;
    int min_size = 0;
    if (string.size() > 0 && (string[0] == '-' || string[0] == '+')) {
        it++;
        min_size++;
    }
    while (it != string.end()) {
        if (*it == '.') {
            if (!decimalPoint) {
                decimalPoint = true;
            } else {
                break;
            }
        } else if (!std::isdigit(*it)) {
            break;
        }
        ++it;
    }
    return string.size() > min_size && it == string.end();
}

bool Model::test_integral(std::string const & string) const {
    std::string::const_iterator it = string.begin();
    int min_size = 0;
    if (string.size() > 0 && (string[0] == '-' || string[0] == '+')) {
        it++;
        min_size++;
    }
    while (it != string.end()) {
        if (!std::isdigit(*it)) { break; }
        ++it;
    }
    return string.size() > min_size && it == string.end();
}

std::set< Bitmask > Model::partitions(void) const {
    std::set< Bitmask > results;
    if (this -> submodels.size() == 0) {
        results.insert(this -> capture);
    } else {
        std::map< bool, Model > const & submodel_map = this -> submodels;
        for (auto iterator = submodel_map.begin(); iterator != submodel_map.end(); ++iterator) {
            std::set<Bitmask> const & subpartitions = (iterator -> second).partitions();
            for (auto subiterator = subpartitions.begin(); subiterator != subpartitions.end(); ++subiterator) {
                results.insert(* subiterator);
            }
        }
    }
    return results;
};

size_t const Model::hash(void) const {
    std::set< Bitmask > const & masks = partitions();
    size_t seed = masks.size();
    for (auto iterator = masks.begin(); iterator != masks.end(); ++iterator) {
        seed ^=  (* iterator).hash() + 0x9e3779b9 + (seed<<6) + (seed>>2);
    }
    return seed;
}

bool const Model::operator==(Model const & other) const {
    if (hash() != other.hash()) {
        return false;
    } else {
        std::set< Bitmask > const & masks = partitions();
        std::set< Bitmask > const & other_masks = other.partitions();
        if (masks.size() != other_masks.size()) { return false; }
        auto iterator = masks.begin();
        auto other_iterator = other_masks.begin();
        while (iterator != masks.end() && other_iterator != other_masks.end()) {
            if ((* iterator) != (* other_iterator)) { return false; }
            ++iterator;
            ++other_iterator;
        }
        return true;
    }
}

std::string Model::predict(Bitmask const & sample) const {
    // Currently a stub, need to implement
    if (this -> submodels.size() == 0) {
        return this -> prediction;
    } else {
        unsigned branch_index = sample[this -> feature];
        Model const & submodel = this-> submodels.at(branch_index);
        return submodel.predict(sample);
    }
}

std::string Model::serialize(void) const {
    return to_json().dump();
}

std::string Model::serialize(int const spacing) const {
    return to_json().dump(spacing);
}

json Model::to_json(void) const {
    json node = json::object();
    if (this -> submodels.size() == 0) {
         // Later need to cast this to num if possible
        if (test_integral(this -> prediction)) {
            node["prediction"] = atoi(this -> prediction.c_str());
        } else if (test_rational(this -> prediction)) {
            node["prediction"] = atof(this -> prediction.c_str());
        } else {
            node["prediction"] = this -> prediction;
        }
        node["name"] = this -> feature_name;
        node["loss"] = this -> loss;
        node["complexity"] = this -> complexity;
    } else {
        node["feature"] = this -> feature;
        node["name"] = this -> feature_name;
        node["relation"] = this -> relation;
        if (test_integral(this -> reference)) {
            node["reference"] = atoi(this -> reference.c_str());
        } else if (test_rational(this -> reference)) {
            node["reference"] = atof(this -> reference.c_str());
        } else {
            node["reference"] = this -> reference;
        }
        std::map< bool, Model > const & submodel_map = this -> submodels;
        for (auto iterator = submodel_map.begin(); iterator != submodel_map.end(); ++iterator) {
            std::string key = iterator -> first ? "true" : "false";
            node[key] = (iterator -> second).to_json();
        }
    }
    return node;
}