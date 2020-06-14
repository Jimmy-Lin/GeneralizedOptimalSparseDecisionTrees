#ifndef KEY_PAIR_H
#define KEY_PAIR_H

#include "key.hpp"

class KeyPair {
public:
    // Constructor
    KeyPair(void);
    KeyPair(Key const & source, Key const & destination);
    KeyPair(KeyPair const & other);
    KeyPair & operator=(KeyPair const & other);

    Key const & source(void) const;
    Key const & destination(void) const;
    int const hash(void) const;

    bool const operator==(KeyPair const & other) const;
    bool const operator!=(KeyPair const & other) const;
private:
    Key _source;
    Key _destination;
    int hash_code;
};

namespace std {
    // Hash implementation
    template <>
    struct hash< KeyPair > {
        std::size_t operator()(KeyPair const & key_pair) const {
            return key_pair.hash();
        }
    };
}

#endif