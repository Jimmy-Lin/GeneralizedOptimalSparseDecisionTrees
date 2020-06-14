void Optimizer::execute(Key const & key) {
    int const m = this -> dataset.width();
    float const epsilon = std::numeric_limits<float>::epsilon();

    // From this point, this task cannot be read or written to by other threads
    task_table::accessor task_accessor;
    if (this -> graph.tasks.find(task_accessor, key) == false) { throw "Failed Access to Task (Read)"; }
    Task & task = task_accessor -> second;

    // if (configuration()["verbose"]) {
    //     std::cout << "Task(" << key.indicator().to_string() << ", " << key.feature_index() << "), Bounds = " << "[" << task.lowerbound() << ", " << task.upperbound() << "], Sensitivity(" << task.sensitivity().to_string() << ")" << " State(E|D|R|C) = " << task.explored() << task.delegated() << task.resolved() << task.cancelled() << std::endl; 
    // }

    // Cancels upward flow of information
    if (task.cancelled() || task.resolved()) { return; }

    if (task.explored() == false) {
        if (key.indicator().size() == key.indicator().count()) {
            std::cout << "Task(" << key.indicator().to_string() << ", " << key.feature_index() << "), Bounds = " << "[" << task.lowerbound() << ", " << task.upperbound() << "], Sensitivity(" << task.sensitivity().to_string() << ")" << " State(E|D|R|C) = " << task.explored() << task.delegated() << task.resolved() << task.cancelled() << std::endl;
        }
        task.explore(); // Mark the task as explored once we complete the expansion

        Bitmask const & sensitivity = task.sensitivity();
        // This groups the currently sensitive features such that each group has a common resulting partition on the data set
        // Only one feature from each group needs to be explored, we refer to this as the representative feature of the group
        // The choosing the representative feature from a group is somewhat arbitrary to the training objective
        // There are some possible tricks that can be done to optimize it for the test objective, but we won't go deep into that
        std::unordered_map< Bitmask, std::vector< unsigned int, tbb::scalable_allocator< unsigned int > >,
            std::hash< Bitmask >,
            std::equal_to< Bitmask >,
            tbb::scalable_allocator< std::pair< Bitmask const, std::vector< unsigned int, tbb::scalable_allocator< unsigned int > > > >
        > feature_groups;
        for (unsigned int j = 0; j < m; ++j) {
            if (!task.sensitive(j)) { continue; }
            auto const & partitions = this -> dataset.partition(key.indicator(), j);
            auto iterator = partitions.begin();
            Bitmask const & negative_key = * iterator;
            ++iterator;
            Bitmask const & positive_key = * iterator;

            if (negative_key.count() == 0 || positive_key.count() == 0) {
                task.desensitize(j);
                continue;
            }
            Bitmask const & feature_group_key = (negative_key <= positive_key) ? negative_key : positive_key;
            feature_groups[feature_group_key].push_back(j); // Records this feature into the corresponding feature group
        }
        // Use the computed feature_groups to select representative features and prune non-representative features
        for (auto iterator = feature_groups.begin(); iterator != feature_groups.end(); ++iterator) {
            auto const & group = iterator -> second;

            unsigned int index_total = 0;
            unsigned int group_size = group.size();
            for (unsigned int i = 0; i < group.size(); ++i) {
                index_total += group[i];
            }
            float index_average = ((float) index_total) / ((float) group_size);
            unsigned int representative_index = * group.begin();
            for (unsigned int i = 0; i < group.size(); ++i) {
                unsigned int feature_index = group[i];
                float distance = std::abs((float) feature_index - index_average);
                float best_distance = std::abs((float) representative_index - index_average);
                if (distance < best_distance) {
                    representative_index = feature_index;
                }
            }

            for (unsigned int i = 0; i < group.size(); ++i) {
                unsigned int feature_index = group[i];
                if (feature_index != representative_index) { task.desensitize(feature_index); }
            }
        }

        // Shrink similarity index
        // for (unsigned int j = 0; j < m; ++j) {
        //     if (task.similarity_index.count(j) == 0) { continue; }
        //     if (!task.sensitive(j)) {
        //         task.similarity_index.erase(j);
        //         continue;
        //     }
        //     auto partition_j = this -> dataset.partition(key.indicator(), j);
        //     Bitmask const & negative_j = * partition_j.begin();
        //     Bitmask const & positive_j = * (++partition_j.begin());

        //     similarity_index_type & index = task.similarity_index[j];
        //     std::vector< unsigned int > erased;
        //     for (auto iterator = index.begin(); iterator != index.end(); ++iterator) {
        //         unsigned int k = iterator -> first;
        //         if (!task.sensitive(k) || j == k) {
        //             erased.push_back(k);
        //             continue;
        //         }
        //         auto partition_k = this -> dataset.partition(key.indicator(), k);
        //         Bitmask const & negative_k = * partition_k.begin();
        //         Bitmask const & positive_k = * (++partition_k.begin());

        //         // Compute the two possible difference sets
        //         Bitmask const & alpha_mask = (negative_j & positive_k) | (positive_j & negative_k);
        //         Bitmask const & beta_mask = (negative_j & negative_k) | (positive_j & positive_k);
        //         // Computes the smaller of the total weight of each mask
        //         // This forms the distance "omega" in the simiar support bound, with awareness of weights modified by preprocessing
        //         float distance = std::min(
        //             this -> dataset.support(alpha_mask),
        //             this -> dataset.support(beta_mask)
        //         );
        //         // Encodes the maximal distance between j and k
        //         index[k] = distance;
        //     }
        //     for (unsigned int k = 0; k < erased.size(); ++k) {
        //         index.erase(erased[k]);
        //     }
        // }

        // For future work, this can be parallelized
        for (unsigned int j = 0; j < m; ++j) {
            if (!task.sensitive(j)) { continue; }
            bitset selector;
            selector.resize(this -> dataset.width(), 1);
            selector.set(j, false);
            Bitmask const reduced_sensitivity = sensitivity & Bitmask(selector);

            auto const & partitions = this -> dataset.partition(key.indicator(), j);
            for (unsigned int k = 0; k < partitions.size(); ++k) {
                Key subkey(partitions[k]);

                if (this -> graph.tasks.count(subkey) == 0) {
                    this -> graph.tasks.insert(std::make_pair(subkey, new_task(subkey, reduced_sensitivity, task.similarity_index))); // Attempt to insert a new task
                } else {
                    this -> graph.hits += 1;
                }

                if (this -> graph.backward_index.count(subkey) == 0) {
                    this -> graph.backward_index.insert(std::make_pair(subkey, index_type())); // Initializes a new index
                }

                index_table::accessor subtask_backward_index_accessor;
                if (this -> graph.backward_index.find(subtask_backward_index_accessor, subkey) == false) { throw "Failed Access to Backward Index (Write)"; }
                // Specifying the feature index allows upward calls to be aware of the particular feature involved
                subtask_backward_index_accessor -> second.insert(Key(key.indicator(), j));
            }
        }

    } // From this point, the task has completed the expansion

    // if (configuration()["verbose"]) {
    //     std::cout << "Explored Task(" << key.indicator().to_string() << ", " << key.feature_index() << "), Bounds = " << "[" << task.lowerbound() << ", " << task.upperbound() << "]" << " State(E|D|R|C) = " << task.explored() << task.delegated() << task.resolved() << task.cancelled() << std::endl; 
    // }

    for (unsigned int j = 0; j < m; ++j) {
        if (!task.sensitive(j)) { continue; }
        // if (key.feature_index() >= 0 && j != key.feature_index()) { continue; }
        std::pair< float, float > & combined_bound = task.combined_bounds[j];
        if (combined_bound.second == 0.0) { combined_bound.second = 1.0; }
        float combined_lowerbound = 0.0;
        float combined_upperbound = 0.0;
        auto const & partitions = this -> dataset.partition(key.indicator(), j);
        bool found = true;
        for (unsigned int k = 0; k < partitions.size(); ++k) {
            Key subkey(partitions[k]);
            task_table::const_accessor subtask_accessor;
            if (this -> graph.tasks.find(subtask_accessor, subkey) == false) {
                // throw "Failed Access to Subtask (Reduction)";
                found = false;
                break;
            }
            Task const & subtask = subtask_accessor -> second;
            combined_lowerbound += subtask.lowerbound();
            combined_upperbound += subtask.upperbound();
            if (found && key.indicator().size() == key.indicator().count() && j == 11) {
            // for (unsigned int k = 0; k < partitions.size(); ++k) {
                // std::cout << this->dataset.support(partitions[k]) * this->dataset.size() << std::endl; 
                // std::cout << partitions[k].to_string() << std::endl;
                std::cout << "Feature " << j << "-" << k << ": [" << subtask.lowerbound() << ", " << subtask.upperbound() << "]" << std::endl;
            }
        }
        if (found && key.indicator().size() == key.indicator().count() && j == 11) {
            for (unsigned int k = 1; k < partitions.size(); ++k) {
                // std::cout << this->dataset.support(partitions[k]) * this->dataset.size() << std::endl; 
                std::cout << partitions[k].to_string() << std::endl;
            }
            std::cout << "Feature " << j << ": [" << combined_lowerbound << ", " << combined_upperbound << "]" << std::endl;
        }
        combined_bound.first = std::max(combined_bound.first, combined_lowerbound);
        combined_bound.second = std::min(combined_bound.second, combined_upperbound);
        // std::cout << "feature " << j << " : [" << combined_bound.first << ", " << combined_bound.second << "]" << std::endl;
    }

    // if (configuration()["verbose"]) {
    //     std::cout << "Summed Task(" << key.indicator().to_string() << ", " << key.feature_index() << "), Bounds = " << "[" << task.lowerbound() << ", " << task.upperbound() << "]" << " State(E|D|R|C) = " << task.explored() << task.delegated() << task.resolved() << task.cancelled() << std::endl; 
    // }

    float const initial_lowerbound = task.lowerbound();
    float const initial_upperbound = task.upperbound();
    float lowerbound = initial_upperbound;
    float upperbound = initial_upperbound;
    // For future work, this can be parallelized

    for (unsigned int j = 0; j < m; ++j) {
        if (!task.sensitive(j)) { continue; }
        // if (key.feature_index() >= 0 && j != key.feature_index()) { continue; }
        std::pair< float, float > & combined_bound = task.combined_bounds[j];
        lowerbound = std::min(combined_bound.first, lowerbound);
        upperbound = std::min(combined_bound.second, upperbound);
    }

    for (unsigned int j = 0; j < m; ++j) {
        if (!task.sensitive(j)) { continue; }
        std::pair< float, float > & combined_bound = task.combined_bounds[j];
        if (combined_bound.first > upperbound + epsilon) { task.desensitize(j); }
    }

    // // Apply the similar support bound specifically to raise lowerbounds
    // for (unsigned int j = 0; j < m; ++j) {
    //     if (!task.sensitive(j)) { continue; }
    //     std::pair< float, float > & target_bound = task.combined_bounds[j];
    //     if (target_bound.first > lowerbound) { continue; }
    //     if (target_bound.first == target_bound.second) { continue; }
    //     auto partition_j = this -> dataset.partition(key.indicator(), j);
    //     Bitmask const & negative_j = *partition_j.begin();
    //     Bitmask const & positive_j = *(++partition_j.begin());
    //     for (unsigned int k = 0; k < m; ++k) {
    //         if (j == k) { continue; }
    //         auto partition_k = this -> dataset.partition(key.indicator(), k);
    //         Bitmask const & negative_k = *partition_k.begin();
    //         Bitmask const & positive_k = *(++partition_k.begin());
    //         // Compute the two possible difference sets
    //         Bitmask const & alpha_mask = (negative_j & positive_k) | (positive_j & negative_k);
    //         Bitmask const & beta_mask = (negative_j & negative_k) | (positive_j & positive_k);
    //         // Computes the smaller of the total weight of each mask
    //         // This forms the distance "omega" in the simiar support bound, with awareness of weights modified by preprocessing
    //         float distance = std::min(
    //             this -> dataset.support(alpha_mask),
    //             this -> dataset.support(beta_mask)
    //         );
    //         if (negative_k == negative_k || negative_k == positive_j) { continue; }
    //         if (this -> graph.tasks.count(negative_k) == 0 || this -> graph.tasks.count(positive_k) == 0) { continue; }

    //         // Calculate reference bound
    //         float reference_lowerbound = 0.0;
    //         float reference_upperbound = 0.0;
    //         for (auto iterator = partition_k.begin(); iterator != partition_k.end(); ++iterator) {
    //             Key subkey(* iterator);
    //             task_table::const_accessor subtask_accessor;
    //             if (this -> graph.tasks.find(subtask_accessor, subkey) == false) { throw "Failed Access to Subtask (Reduction)"; }
    //             Task const & subtask = subtask_accessor -> second;
    //             reference_lowerbound += subtask.lowerbound();
    //             reference_upperbound += subtask.upperbound();
    //         }
    //         float similar_lowerbound = std::max(target_bound.first, reference_lowerbound - distance);
    //         float similar_upperbound = std::min(target_bound.second, reference_upperbound + distance);
    //         // std::cout << "Distance[" << j << ", " << k << "] = " << distance << std::endl;
    //         // std::cout << "PriorBound(" << task.combined_bounds[k].first << " ," << task.combined_bounds[k].second << ")" << std::endl;
    //         std::cout << "SimilarSupportBound(" << similar_lowerbound << " ," << similar_upperbound << ")" << std::endl;
    //         // task.combined_bounds[j].first = similar_lowerbound;
    //         // task.combined_bounds[j].second = similar_upperbound;
    //     }
    // }

    // if (configuration()["verbose"]) {
    //     std::cout << "Minimized Task(" << key.indicator().to_string() << ", " << key.feature_index() << "), Bounds = " << "[" << task.lowerbound() << ", " << task.upperbound() << "]" << " State(E|D|R|C) = " << task.explored() << task.delegated() << task.resolved() << task.cancelled() << std::endl; 
    // }

    // While the strong upperbound represents the bound of the best global optima so far
    // The weak upperbound represents the bound of all global optima, which tends to lag behind by a little bit
    // The strong upperbound is used to represent how promising the task is, but the weak upperbound is used to decide termination
    float complete_upperbound = upperbound;
    // For future work, this can be parallelized
    for (unsigned int j = 0; j < m; ++j) {
        if (!task.sensitive(j)) { continue; }
        std::pair< float, float > & combined_bound = task.combined_bounds[j];
        float combined_lowerbound = combined_bound.first;
        float combined_upperbound = combined_bound.second;

        // std::cout << "Task.combined_bounds[" << j << "] = (" << combined_lowerbound << ", " << combined_upperbound << ")" << std::endl; 

        if (combined_lowerbound <= upperbound + epsilon) {
            complete_upperbound = std::max(combined_upperbound, complete_upperbound);
        }
    }
    float scope = std::min(upperbound, task.scope());
    // upperbound = complete_upperbound;
    // upperbound = scope;
    // std::cout << "Task(" << key.indicator().to_string() << ")::inform(" << lowerbound << ", " << upperbound << ")" << std::endl;
    task.inform(lowerbound, upperbound);

     // Task Complete
    if ( complete_upperbound - lowerbound < epsilon ) {
        // std::cout << key.indicator().to_string() << " resolved: [" << lowerbound << ", " << complete_upperbound << "]" << std::endl;

        task.resolve();
    }
    if (lowerbound != initial_lowerbound || upperbound != initial_upperbound) {
        async_return(key, INFORMATION_PRIORITY, 1 - task.support(), task.priority(this->optimism)); // Inform parents of incremental updates
    }

    // From this point, we know that further exploration of subtasks needs to be done
    if (task.delegated() == false && task.resolved() == false) {
        task.delegate();
        // std::cout << key.indicator().to_string() << " bounded by " << upperbound << std::endl;
        for (unsigned int j = 0; j < m; ++j) {
            if (!task.sensitive(j)) { continue; }
            std::pair< float, float > & combined_bound = task.combined_bounds[j];
            float combined_lowerbound = combined_bound.first;
            float combined_upperbound = combined_bound.second;
            if (combined_lowerbound == combined_upperbound) { continue; }
            if (combined_lowerbound <= upperbound) {
                auto const & partitions = this -> dataset.partition(key.indicator(), j);
                for (unsigned int k = 0; k < partitions.size(); ++k) {
                    Key subkey(partitions[k]);
                    task_table::accessor subtask_accessor;
                    if (this -> graph.tasks.find(subtask_accessor, subkey) == false) { throw "Failed Access to Subtask (Reduction)"; }
                    Task & subtask = subtask_accessor -> second;
                    float complement_lowerbound = combined_lowerbound - task.lowerbound();
                    float local_scope = scope - complement_lowerbound; // How poorly this subproblem can perform while remaining relevant to the parent

                    if (subtask.resolved()) { continue; }
                    // if (subtask.scope() == 1.0) {
                    //     subtask.rescope(local_scope);
                        async_call(subkey, DELEGATION_PRIORITY, 1 - subtask.support(), subtask.priority(this -> optimism));
                    // } else if (subtask.scope() < local_scope) {
                    //     subtask.rescope(local_scope);
                    //     async_call(subkey, RESCOPE_PRIORITY, subtask.support(), subtask.priority(this -> optimism));
                    // }
                    // if (configuration()["verbose"]) {
                    //     std::cout << "Delegated Subtask(" << subkey.indicator().to_string() << ", " << subkey.feature_index() << "), Bounds = " << "[" << subtask.lowerbound() << ", " << subtask.upperbound() << "]" << " State(E|D|R|C) = " << subtask.explored() << subtask.delegated() << subtask.resolved() << subtask.cancelled() << std::endl; 
                    // }
                }
            }
        }
    }

    return;
}