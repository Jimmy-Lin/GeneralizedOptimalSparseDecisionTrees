#ifndef CONFIGURATION_H
#define CONFIGURATION_H

#include <iostream>
#include <json/json.hpp>
#include "reference.hpp"

using json = nlohmann::json;

// Static configuration object used to modifie the algorithm behaviour
// By design, all running instances of the algorithm within the same process must share the same configuration
class Configuration {
public:
    static void configure(std::istream & configuration);
    static void configure(json source);
    static std::string to_string(unsigned int spacing = 0);

    static float uncertainty_tolerance; // The maximum allowed global optimality before the optimization can terminate
    static float regularization; // The penalty incurred for each leaf in the model
    static bool allow_small_reg; // Flag for allowing regularization < 1/n , where n = num_samples (if false, regularizations below 1/n are automatically set to 1/n)
    static float upperbound; // Upperbound on the root problem for pruning problems using a greedy model

    static unsigned int time_limit; // The maximum allowed runtime (seconds). 0 means unlimited.
    static unsigned int worker_limit; // The maximum allowed worker threads. 0 means match number of available cores
    static unsigned int stack_limit; // The maximum amount of stack space (bytes) allowed to use as buffers
    static unsigned int precision_limit; // The maximum number of significant figures considered for each ordinal feature
    static unsigned int model_limit; // The maximum number of models extracted

    static bool verbose; // Flag for printing status to standard output
    static bool diagnostics; // Flag for printing diagnosis to standard output if a bug is detected

    static unsigned char depth_budget; // The maximum tree depth for solutions, counting a tree with just the root node as depth 1. 0 means unlimited.
    static bool reference_LB; // Flag for using a vector of misclassifications from another (reference) model to lower bound our own misclassifications
    static std::string path_to_labels; //if reference_LB is true, gives file path to the labels from the reference model. Otherwise, not used
    //if reference lb is true, configure instantiates the Reference class with the appropriate labels

    static bool balance; // Flag for adjusting the importance of each row to equalize the total importance of each class (overrides weight)
    static bool look_ahead; // Flag for enabling the one-step look-ahead bound implemented via scopes
    static bool similar_support; // Flag for enabling the similar support bound imeplemented via the distance index
    static bool cancellation; // Flag for enabling upward propagation of cancelled subproblems
    static bool continuous_feature_exchange; // Flag for enabling the pruning of neighbouring thresholds using subset comparison
    static bool feature_exchange; // Flag for enabling the pruning of pairs of features using subset comparison
    static bool feature_transform; // Flag for enabling the equivalence discovery through simple feature transformations
    static bool rule_list; // Flag for enabling rule-list constraints on models
    static bool non_binary; // Flag for enabling non-binary encoding

    static std::string costs; // Path to file containing cost matrix
    static std::string model; // Path to file used to store the extracted models
    static std::string timing; // Path to file used to store training time
    static std::string trace; // Path to directory used to store traces
    static std::string tree; // Path to directory used to store tree-traces
    static std::string profile; // Path to file used to log runtime statistics
};

#endif
