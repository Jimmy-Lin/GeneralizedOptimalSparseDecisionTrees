#include "message.hpp"

Message::Message(void){};

Message::~Message(void) {};

void Message::initialize(unsigned int height, unsigned int width, unsigned int depth) {
    this -> sender_tile.resize(height * (width + depth));
    this -> recipient_tile.resize(height * (width + depth));
    this -> recipient_capture.resize(height);
    this -> recipient_feature.resize(width);
    this -> features.resize(width);
    this -> signs.resize(width);
}

void Message::exploration(Tile const & sender, Bitmask const & recipient_capture, Bitmask const & recipient_feature, int feature, float scope, float primary, float secondary, float tertiary) {
    this -> sender_tile = sender;
    this -> recipient_capture = recipient_capture;
    this -> recipient_feature = recipient_feature;
    
    if (feature != 0) {
        this -> features.clear();
        this -> features.set(std::abs(feature) - 1, true);
        this -> signs.clear();
        this -> signs.set(std::abs(feature) - 1, feature > 0);
    }

    this -> scope = scope;

    this -> code = Message::exploration_message;

    this -> _primary = primary;
    this -> _secondary = secondary;
    this -> _tertiary = tertiary;
}

void Message::exploitation(Tile const & sender, Tile const & recipient, Bitmask const & features, float primary, float secondary, float tertiary) {
    this -> sender_tile = sender;
    this -> recipient_tile = recipient;
    
    this -> features = features;
    this -> code = Message::exploitation_message;

    this -> _primary = primary;
    this -> _secondary = secondary;
    this -> _tertiary = tertiary;
}

Message & Message::operator=(Message const & other) {
    this -> sender_tile = other.sender_tile;
    this -> recipient_tile = other.recipient_tile;
    this -> recipient_capture = other.recipient_capture;
    this -> recipient_feature = other.recipient_feature;
    this -> feature = other.feature;
    this -> features = other.features;
    this -> signs = other.signs;
    this -> scope = other.scope;
    this -> code = other.code;
    this -> _primary = other._primary;
    this -> _secondary = other._secondary;
    this -> _tertiary = other._tertiary;
    return * this;
};

bool Message::operator<(Message const & other) const {
    if (this -> _primary != other._primary) {
        return this -> _primary < other._primary;
    } else if (this -> _secondary != other._secondary) {
        return this -> _secondary < other._secondary;
    } else if (this -> _tertiary != other._tertiary) {
        return this -> _tertiary < other._tertiary;
    }
    return false;
}

bool Message::operator>(Message const & other) const {
    if (this -> _primary != other._primary) {
        return this -> _primary > other._primary;
    } else if (this -> _secondary != other._secondary) {
        return this -> _secondary > other._secondary;
    } else if (this -> _tertiary != other._tertiary) {
        return this -> _tertiary > other._tertiary;
    }
    return false;
}

bool Message::operator<=(Message const & other) const {
    if (this -> _primary != other._primary) {
        return this -> _primary < other._primary;
    } else if (this -> _secondary != other._secondary) {
        return this -> _secondary < other._secondary;
    } else if (this -> _tertiary != other._tertiary) {
        return this -> _tertiary < other._tertiary;
    }
    return true;
}

bool Message::operator>=(Message const & other) const {
    if (this -> _primary != other._primary) {
        return this -> _primary > other._primary;
    } else if (this -> _secondary != other._secondary) {
        return this -> _secondary > other._secondary;
    } else if (this -> _tertiary != other._tertiary) {
        return this -> _tertiary > other._tertiary;
    }
    return true;
}

bool Message::operator==(Message const & other) const {
    if (this -> code != other.code) { return false; }
    switch (this -> code) {
        case Message::exploration_message: {
            return this -> sender_tile == other.sender_tile
                && this -> recipient_capture == other.recipient_capture;
                // && this -> features == other.features
                // && this -> signs == other.signs
                // && this -> scope == other.scope;
            break;
        }
        case Message::exploitation_message: {
            // return this -> features == other.features
            //     && this -> recipient_tile == other.recipient_tile;

            return this -> recipient_tile == other.recipient_tile;
            break;
        }
        default: {
            return false;
            break;
        }
    }
}

size_t Message::hash(void) const {
    size_t seed = 0;
    switch (this -> code) {
        case Message::exploration_message: {
            seed ^= this -> sender_tile.hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= this -> recipient_capture.hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            // seed ^= this -> feature + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            break;
        }
        case Message::exploitation_message: {
            // seed ^= this -> features.hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            seed ^= this -> recipient_tile.hash() + 0x9e3779b9 + (seed << 6) + (seed >> 2);
            break;
        }
        default: {
            break;
        }
    }
    return seed;
}