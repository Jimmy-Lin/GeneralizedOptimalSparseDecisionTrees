#ifndef REFERENCE_H
#define REFERENCE_H

#include <vector>

#include "bitmask.hpp"
#include "encoder.hpp"

class Reference {
public: 
    static void initialize_labels(std::istream & labels);

    //labels for each row of the dataset, according to the reference model
    //Will likely include misclassifications when compared to the true labels
    static std::vector<Bitmask> labels;
};

#endif
