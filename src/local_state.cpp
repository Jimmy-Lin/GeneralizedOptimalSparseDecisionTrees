#include "local_state.hpp"

LocalState::LocalState(void) {}

void LocalState::initialize(unsigned int _samples, unsigned int _features, unsigned int _targets) {
    this -> samples = _samples;
    this -> features = _features;
    this -> targets = _targets;

    this -> inbound_message.initialize(_samples, _features, _targets);
    this -> outbound_message.initialize(_samples, _features, _targets);
    
    this -> neighbourhood.resize(2 * (this -> features));

    unsigned int buffer_count = 4;
    unsigned int row_size = this -> features + this -> targets;
    unsigned int column_size = this -> samples;
    unsigned int max_tile_size = row_size * column_size;

    for (unsigned int i = 0; i < buffer_count; ++i) {
        this -> rows.emplace_back(row_size);
        this -> columns.emplace_back(column_size);
    }
}

LocalState & LocalState::operator=(LocalState const & source) {
    this -> neighbourhood = source.neighbourhood;
    this -> rows = source.rows;
    this -> columns = source.columns;
    return * this;
}


LocalState::~LocalState(void) {
    this -> neighbourhood.clear();
    this -> rows.clear();
    this -> columns.clear();
}
