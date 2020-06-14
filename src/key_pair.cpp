#include "key_pair.hpp"

KeyPair::KeyPair(void) {}

KeyPair::KeyPair(Key const & source, Key const & destination) : _source(source), _destination(destination) {
    size_t seed = 0;
    seed ^=  std::hash< Bitmask >()(source.indicator()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    seed ^=  std::hash< Bitmask >()(destination.indicator()) + 0x9e3779b9 + (seed<<6) + (seed>>2);
    this -> hash_code = seed;
    return;
}

KeyPair::KeyPair(KeyPair const & other) : _source(other._source), _destination(other._destination), hash_code(other.hash_code) {}

KeyPair & KeyPair::operator=(KeyPair const & other) {
    this -> _source = other._source;
    this -> _destination = other._destination;
    this -> hash_code = other.hash_code;
    return * this;
}

Key const & KeyPair::source(void) const {
    return this -> _source;
}

Key const & KeyPair::destination(void) const {
    return this -> _destination;
}

int const KeyPair::hash(void) const {
    return this -> hash_code;
}

bool const KeyPair::operator==(KeyPair const & other) const {
    // return (this -> _source == other._source) && (this -> _destination == other._destination);
    if (false) {
        return false;
    // } else if (this -> hash_code != other.hash_code) {
    //     return false; // Early exit based on hash difference
    // } else if (this -> _destination.indicator().hash() != other._destination.indicator().hash() || this -> _source.indicator().hash() != other._source.indicator().hash()) {
    //     return false; // Early exit based on hash difference of either component's indicator
    // } else if (this -> _destination.feature_index() != other._destination.feature_index() || this -> _source.feature_index() != other._source.feature_index()) {
    //     return false; // Early exit based on hash difference of either component's feature index
    // } else if (this -> _destination.indicator().encoding().size() != other._destination.indicator().encoding().size() || this -> _source.indicator().encoding().size() != other._source.indicator().encoding().size()) {
    //     return false; // Early exit based on encoding size difference
    } else {
        return (this -> _source == other._source) && (this -> _destination == other._destination);
    }
}

bool const KeyPair::operator!=(KeyPair const & other) const {
    return !((* this) == other);
}