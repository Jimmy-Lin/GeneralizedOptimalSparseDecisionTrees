#include "reference.hpp"

std::vector<Bitmask> Reference::labels = std::vector<Bitmask>();

void Reference::initialize_labels(std::istream & labels){
    //read labels
    Encoder encoder(labels);
    std::vector<Bitmask>rows = encoder.read_binary_rows();

    // set up Reference::labels to be a vector of bitmasks, 
    // where each bitmask represents one of the columns of the binarized labels
    // (similar to code used in Dataset::construct_bitmasks - may need to make a helper fn common to both classes)
    unsigned int number_of_binary_targets = encoder.binary_targets(); // Number of target features
    unsigned int number_of_samples = encoder.samples();
    for (unsigned int i = 0; i < number_of_binary_targets; ++i) {Reference::labels.emplace_back(number_of_samples); }
    for (unsigned int i = 0; i < number_of_samples; ++i) {
        for (unsigned int j = 0; j < number_of_binary_targets; ++j) {
            Reference::labels[j].set(i, bool(rows[i][j]));
        }
    }
};
