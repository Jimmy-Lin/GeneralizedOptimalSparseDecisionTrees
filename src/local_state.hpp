#ifndef LOCAL_STATE_H
#define LOCAL_STATE_H

class LocalState;

#include "bitmask.hpp"
#include "message.hpp"
#include "task.hpp"
#include "tile.hpp"

// Container of all data structures the local state owned by each thread
class LocalState {
public:
    LocalState(void);
    ~LocalState(void);

    // @param samples: The number of samples a column bit mask has to represent
    // @param features: The number of features a row bit mask has to represent
    // @param targets: The number of targets a row bit mask has to represent
    // @note: bit masks are used to represent rows and columns of a data set
    //        samples refer to the number of independent samples in a dataset
    //        features refer to the number of binary features that will be available at prediction time
    //        targets refer to the number of different classes that a sample can fall under
    void initialize(unsigned int samples, unsigned int features, unsigned int targets);
    
    LocalState & operator=(LocalState const & source);

    std::vector< Task > neighbourhood; // Memory buffer for storing children of a node
    Message inbound_message; // Memory buffer for storing a messages from the priority queue
    Message outbound_message; // Memory buffer for storing a messages from the priority queue
    // std::vector< Tile > keys; // Memory buffer for storing a tile representation of a node identifier
    std::vector< Bitmask > rows; // Memory buffer for storing a bitmask representation of a feature set + target set
    std::vector< Bitmask > columns; // Memory buffer for storing a bitmask representation of a capture set

    // Bitmask dirty; // Mask indicating which items in the neighbourhood needs to be written?
    
    unsigned int samples;
    unsigned int features;
    unsigned int targets;
};

#endif