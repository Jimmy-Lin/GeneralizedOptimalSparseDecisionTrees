Task Optimizer::new_task(Key const & key, Bitmask const & sensitivity, similarity_index_table_type const & parent_similarity_index) {
    float const epsilon = std::numeric_limits<float>::epsilon();
    auto leaf = this -> dataset.leaf(key.indicator());
    float const regularization = this -> regularization;
    float const lowerbound = this -> dataset.loss_lowerbound(key.indicator()) + regularization;
    float const upperbound = std::get<1>(leaf) + regularization;
    float const uncertainty = upperbound - lowerbound;
    float const support = this -> dataset.support(key.indicator());
    float const base_objective = std::get<1>(leaf) + regularization;
    if (
        ( uncertainty <= regularization - epsilon ) // Incremental Accuracy Lowerbound Bound Failed
        || ( 0.5 * support <= regularization - epsilon ) // Support Lowerbound Failed
        || ( key.indicator().count() <= 1 ) // Equivalent Group cannot be split further
        || ( sensitivity.count() <= 0 ) // Equivalent Group cannot be split further
    ) {
        Task task(lowerbound,
            upperbound,
            support,
            base_objective, 
            sensitivity, 
            parent_similarity_index);
        task.resolve();
        return task;
    } else if (this -> sample_depth > 1 && key.indicator().size() == key.indicator().count()) {
        // Attempt to improve the upperbound by better sampling
        Task task(
            lowerbound + regularization,
            std::min(upperbound, sample(key, sensitivity, this -> sample_depth)),
            support,
            base_objective,
            sensitivity,
            parent_similarity_index);
        return task;
    } else {
        return Task(lowerbound + regularization, 
            upperbound, 
            support, 
            base_objective, 
            sensitivity, 
            parent_similarity_index);
    }
}

float const Optimizer::sample(Key const & key, Bitmask const & sensitivity, int const depth) {
    float const epsilon = std::numeric_limits<float>::epsilon();
    auto leaf = this -> dataset.leaf(key.indicator());
    float const regularization = this -> regularization;
    float const lowerbound = this -> dataset.loss_lowerbound(key.indicator()) + regularization;
    float const upperbound = std::get<1>(leaf) + regularization;
    float const uncertainty = upperbound - lowerbound;
    float const support = this -> dataset.support(key.indicator());
    float const base_objective = std::get<1>(leaf) + regularization;
    if (
        ( uncertainty <= regularization - epsilon ) // Incremental Accuracy Lowerbound Bound Failed
        || ( 0.5 * support <= regularization - epsilon ) // Support Lowerbound Failed
        || ( key.indicator().count() <= 1 ) // Equivalent Group cannot be split further
        || ( sensitivity.count() <= 0 ) // Equivalent Group cannot be split further
        || ( depth <= 1 ) // Max Depth for sampling
    ) {
        // Base Case: Detected that there exists no worthwhile split from now on
        return base_objective;
    } else {
        // Choose a greedy split based on some heuristic to sample this space

        // Calculate current entropy
        float base_entropy = this -> dataset.entropy(key.indicator());

        int index = 0;
        float best_score = 0;
        for (int j = 0;  j < this -> dataset.width(); ++j) {
            if (sensitivity[j] == 0) { continue; }
            auto partitions = this -> dataset.partition(key.indicator(), j);
            float split_entropy = 0;
            for (auto iterator = partitions.begin(); iterator != partitions.end(); ++iterator) {
                split_entropy += this -> dataset.entropy(* iterator);
            }
            float information_gain = base_entropy - split_entropy;
            if (information_gain > best_score) {
                best_score = information_gain;
                index = j;
            }
        }

        float objective = 0.0;

        bitset selector;
        selector.resize(this -> dataset.width(), 1);
        selector[index] = 0;
        Bitmask const reduced_sensitivity = sensitivity & Bitmask(selector);

        auto partitions = this -> dataset.partition(key.indicator(), index);

        for (unsigned int k = 0; k < partitions.size(); ++k) {
            objective += sample(partitions[k], reduced_sensitivity, depth - 1);
        }
        return objective;
    }
}


void Optimizer::async_call(Key const & callee, float const primary_priority, float const secondary_priority, float const tertiary_priority) {
    this -> queue.push(callee, primary_priority, secondary_priority, tertiary_priority);
}

void Optimizer::async_return(Key const & callee, float const primary_priority, float const secondary_priority, float const tertiary_priority) {
    // From this point this index cannot be read or written to by other threads
    index_table::const_accessor backward_index_accessor;
    if (this -> graph.backward_index.find(backward_index_accessor, callee) == false) { throw "Failed Access to Backward Index (Read)"; }
    index_type const & backward_index = backward_index_accessor -> second;
    // For future work, this can be parallelized
    for (auto iterator = backward_index.begin(); iterator != backward_index.end(); ++ iterator) {
        Key const & superkey = * iterator;
        this -> queue.push(superkey, primary_priority, secondary_priority, tertiary_priority);
    }
}