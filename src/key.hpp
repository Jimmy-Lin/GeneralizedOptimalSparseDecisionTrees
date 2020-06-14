#ifndef KEY_H
#define KEY_H

#include "bitmask.hpp"

// Hash implementation

class Key {
public:
    // Constructor
    Key(void);
    Key(Bitmask const & indicator, int const feature_index = -1);
    Key(Key const & other);
    Key & operator=(Key const & other);

    Bitmask const & indicator(void) const;
    int const feature_index(void) const;

    bool const operator==(Key const & other) const;
    bool const operator!=(Key const & other) const;
    bool const operator<(Key const & other) const;
    bool const operator>(Key const & other) const;
    bool const operator<=(Key const & other) const;
    bool const operator>=(Key const & other) const;

private:
    Bitmask _indicator;
    int _feature_index;
};

namespace std {
    // Hash implementation
    template <>
    struct hash< Key > {
        std::size_t operator()(Key const & key) const {
            return std::hash< Bitmask >()(key.indicator());
        }
    };
}

#endif