#ifndef TILE_H
#define TILE_H

#include <iostream>
#include <sstream>

#include "bitmask.hpp"

// Container for tiles which represent an equivalence class of data sets
class Tile {
public:
    // @param content: A bitmask containing the bits of a binary matrix in linearized format
    // @param width: The width of the matrix, used for delinearization
    Tile(Bitmask const & content, unsigned int width);

    // @param samples: an indicator of samples that this tile must capture
    // @param features: an indicator of features that this tile must capture
    // @param id: The id of the local state used when a buffer is needed
    Tile(Bitmask const & samples, Bitmask const & features, unsigned int id);
    Tile(void);
    ~Tile(void);
    
    // Assignment operator used to transfer ownership of data
    Tile & operator=(Tile const & other);

    // Comparison operators used to match different tiles
    bool operator==(Tile const & other) const;
    bool operator!=(Tile const & other) const;

    // Accessors used to inspect/modify the content
    Bitmask & content(void);
    void content(Bitmask const & _new_content);
    unsigned int width(void) const;
    void width(unsigned int _new_width);

    size_t hash(void) const;

    unsigned int size(void) const;
    void resize(unsigned int new_size);

    std::string to_string(void) const;

private:
    Bitmask _content;
    unsigned int _width;
};

// Overrides for STD containers
namespace std {
    template <>
    struct hash< Tile > {
        std::size_t operator()(Tile const & tile) const { return tile.hash(); }
    };

    template <>
    struct equal_to< Tile > {
        bool operator()(Tile const & left, Tile const & right) const { return left == right; }
    };
}

#endif