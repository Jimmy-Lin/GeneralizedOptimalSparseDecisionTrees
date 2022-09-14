#ifndef TASK_H
#define TASK_H

#include <vector>
#include <tuple>
#include <sstream>
#include <mutex>
#include <random>
#include <iostream>
#include <limits>
#include <tbb/scalable_allocator.h>

class Task;

#include "bitmask.hpp"
#include "configuration.hpp"
#include "dataset.hpp"
//#include "graph.hpp" // FIREWOLF: Circular references: Moved to cpp.
#include "integrity_violation.hpp"
#include "queue.hpp"
//#include "state.hpp" // FIREWOLF: Circular references: Moved to cpp.
#include "types.hpp"

class Task {
public:
    Task(void);

    // @param capture_set: indicator for which data points are captured
    // @param feature_set: indicator for which features are still active
    Task(Bitmask const & capture_set, Bitmask const & feature_set, unsigned int id);

    // @returns the support of the this task
    float support(void) const;

    // @returns the objective lowerbound of this task
    float lowerbound(void) const;

    // @returns the lowerbound, without allowing the use of guesses for the lower bound 
    // (which could be overestimates). Currently, only differs from lowerbound() 
    // if Configuration::warm_LB is true. 
    double guaranteed_lowerbound(void);

    // @return the objective upperbound of this task
    float upperbound(void) const;

    float lowerscope(void) const;
    float upperscope(void) const;
    void scope(float new_scope);

    // @return the objective optimality gap of this task
    float uncertainty(void) const;

    // @return the objective risk of not splitting
    float base_objective(void) const;

    // @return the Alkaike information of the captured data
    float information(void) const;

    // @return a bitmask representing the points captured by this task
    Bitmask const & capture_set(void) const;

    // @return a bitmask representing the features that are not yet pruned
    Bitmask const & feature_set(void) const;

    Tile & identifier(void);
    Tile & parent(void);
    std::vector<int> & order(void);

    // @modifies: prunes features
    void prune_feature(unsigned int id);

    // @modifies: inserts children into the cache based on the currently non-pruned features
    void create_children(unsigned int id);

    // @modifies: prunes features
    void prune_features(unsigned int id);

    // @modifies: prunes features based on the indifference bound within adjacent thresholds of ordinal features
    void continuous_feature_exchange(unsigned int id);

    // @modifes: prunes features based on the indifference bound for all feature pairs
    void feature_exchange(unsigned int id);

    void send_explorers(float scope, unsigned int id);

    void send_explorer(Task const & child, float scope, int feature, unsigned int id);

    bool update(float lower, float upper, int optimal_feature);

    // observer method used for debugging
    std::string inspect(void) const;
private:
    Tile _identifier;
    Bitmask _capture_set;
    Bitmask _feature_set;

    std::vector<int> _order;

    float _support;
    float _base_objective;
    float _information;

    float _lowerbound = -std::numeric_limits<float>::max();
    float _upperbound = std::numeric_limits<float>::max();

    // When Configuration::reference_LB is true, _lowerbound is no longer a provable lower bound
    // we use the below variable to track a provable lower bound in this case. 
    float _guaranteed_lowerbound = -std::numeric_limits<float>::max(); 

    float _context_lowerbound = 0.0;
    float _context_upperbound = 0.0;

    float _lowerscope = -std::numeric_limits<float>::max();
    float _upperscope = std::numeric_limits<float>::max();
    float _coverage = -std::numeric_limits<float>::max();

    int _optimal_feature = -1; // Feature index set if part of the oracle model
};

#endif