#include "key.hpp"

Key::Key(void) {}

Key::Key(Bitmask const & indicator, int const feature_index) : _indicator(indicator), _feature_index(feature_index) {}

Key::Key(Key const & other) : _indicator(other._indicator), _feature_index(other._feature_index) {}

Key & Key::operator=(Key const & other) {
    this -> _indicator = other._indicator;
    this -> _feature_index = other._feature_index;
    return * this;
};

Bitmask const & Key::indicator(void) const {
    return this -> _indicator;
}
int const Key::feature_index(void) const {
    return this -> _feature_index;
}

bool const Key::operator==(Key const & other) const {
    return indicator() == other.indicator();
}

bool const Key::operator!=(Key const & other) const {
    return indicator() != other.indicator();
}

bool const Key::operator<(Key const & other) const {
    return indicator() < other.indicator();
}

bool const Key::operator>(Key const & other) const {
    return indicator() > other.indicator();
}

bool const Key::operator<=(Key const & other) const {
    return operator<(other) || operator==(other);
}

bool const Key::operator>=(Key const & other) const {
    return operator>(other) || operator==(other);
}