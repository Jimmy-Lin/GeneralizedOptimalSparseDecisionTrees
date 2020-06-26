#ifndef STATE_H
#define STATE_H

class State;

#include "dataset.hpp"
#include "graph.hpp"
#include "queue.hpp"
#include "local_state.hpp"

// Container of all data structures capturing the state of the optimization
// Here we separate the memory used by the algorithm into two spaces: Global and Local
// Global space is memory that all threads have access to, but is either read-only or is protected by locking mechanisms
// Local space is memory that is partitioned components such each thread has unrestriced access but only to one component

// Local space acts as an "extension" of the stack in a sense that the stack memory semantically belongs to a particular thread.
// The actual location is stored on heap, although hypothetically we can store this on the stack

class State {
public:

    // Global state to which all thread shares access 
    static Dataset dataset;
    static Graph graph; 
    static Queue queue;
    static int status;

    // Local state to which each thread has exclusive access to a single entry
    static std::vector< LocalState > locals;

    static void initialize(std::istream & data_source, unsigned int workers = 1);
    static void reset(void);
};

#endif