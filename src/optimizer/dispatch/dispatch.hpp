bool Optimizer::dispatch(Message const & message, unsigned int id) {
    bool global_update = false;
    switch (message.code) {
        case Message::exploration_message: {
            
            // A message travelling downward in the dependency graph
            Tile const & parent = message.sender_tile;               // The points captured
            Bitmask const & capture_set = message.recipient_capture; // The points captured
            Bitmask const & feature_set = message.recipient_feature; // The features (before pruning)
            bool is_root = capture_set.count() == capture_set.size();
            Task task(capture_set, feature_set, id); // A vertex to represent the problem
            task.scope(message.scope);
            task.create_children(id); // Populate the thread's local cache with child instances
            if (Configuration::feature_exchange || Configuration::continuous_feature_exchange) { task.prune_features(id); } // Prune using a set of bounds
            translation_type order;
            State::dataset.tile(task.capture_set(), task.feature_set(), task.identifier(), task.order(), id);


            vertex_accessor vertex;
            bool inserted = store_self(task.identifier(), task, vertex);

            store_children(vertex -> second, id);

            if (is_root) { // Update the optimizer state
                // float root_upperbound = this -> cart(vertex -> second.capture_set(), vertex -> second.feature_set(), id);
                // std::cout << "Cart: " << root_upperbound << std::endl;
                float root_upperbound = 1.0;
                if (Configuration::upperbound > 0.0) { root_upperbound = std::min(root_upperbound, Configuration::upperbound); }
                vertex -> second.update(vertex -> second.lowerbound(), root_upperbound, -1);
                this -> root = vertex -> second.identifier();
                this -> translator = vertex -> second.order();
                global_update = update_root(vertex -> second.lowerbound(), vertex -> second.upperbound());
            } else { // Connect and signal parents
                adjacency_accessor parents;
                link_to_parent(parent, message.features, message.signs, message.scope, vertex -> second.identifier(), vertex -> second.order(), parents);
                signal_exploiters(parents, vertex -> second, id);
            }

            if (Configuration::reference_LB || message.scope >= vertex -> second.upperscope()) {
                vertex -> second.send_explorers(message.scope, id);
            }

            break;
        }
        case Message::exploitation_message: {
            Tile const & identifier = message.recipient_tile;
            vertex_accessor vertex, left, right;

            load_self(identifier, vertex);

            if (vertex -> second.uncertainty() == 0 || (!Configuration::reference_LB && vertex -> second.lowerbound() >= vertex -> second.upperscope() - std::numeric_limits<float>::epsilon())) { break; }
            bool update = load_children(vertex -> second, message.features, id);

            // if (!update) { break; } // XXX Please check if this check still applies 

            bool is_root = vertex -> second.capture_set().count() == vertex -> second.capture_set().size();
            if (is_root) { // Update the optimizer state
                global_update = update_root(vertex -> second.lowerbound(),  vertex -> second.upperbound());
            } else {
                adjacency_accessor parents; // find backward look-up entry
                load_parents(identifier, parents);
                signal_exploiters(parents, vertex -> second, id); // Signal parents
            }

            break;
        }
        default: {
            std::stringstream reason;
            reason << "Unsupported Message Type: " << message.code;
            throw IntegrityViolation("Optimizer::dispatch", reason.str());
        }
    }
    return global_update;
}

bool Optimizer::load_children(Task & task, Bitmask const & signals, unsigned int id) {
    float lower = task.base_objective(), upper = task.base_objective();
    int optimal_feature = -1;
    bound_accessor bounds;
    State::graph.bounds.find(bounds, task.identifier());
    for (bound_iterator iterator = bounds -> second.begin(); iterator != bounds -> second.end(); ++iterator) {
        int feature = std::get<0>(* iterator);

        if (signals.get(feature)) { // An update is pending
            bool ready = true;
            for (int k = 0; k < 2; ++k) {
                vertex_accessor child;
                child_accessor key;
                ready = ready && State::graph.children.find(key, std::make_pair(task.identifier(), k ?  -(feature + 1) : (feature + 1)))
                    && State::graph.vertices.find(child, key -> second);
                if (ready) {
                    State::locals[id].neighbourhood[2 * feature + k] = child -> second;
                }
            }

            if (ready) {
                float split_lower, split_upper;
                Task const & left = State::locals[id].neighbourhood[2 * feature];
                Task const & right = State::locals[id].neighbourhood[2 * feature + 1];

                if (Configuration::rule_list) {
                    float lower_negative = left.lowerbound() + right.base_objective();
                    float lower_positive = left.base_objective() + right.lowerbound();
                    split_lower = std::min(lower_negative, lower_positive);
                    float upper_negative = left.upperbound() + right.base_objective();
                    float upper_positive = left.base_objective() + right.upperbound();
                    split_upper = std::min(upper_negative, upper_positive);
                } else {
                    split_lower = left.lowerbound() + right.lowerbound();
                    split_upper = left.upperbound() + right.upperbound();
                }

                std::get<1>(* iterator) = split_lower;
                std::get<2>(* iterator) = split_upper;
            }
        }

        if (Configuration::similar_support) {
            if (iterator != bounds -> second.begin()) { // Comparison with previous feature
                unsigned int i, j;
                float j_lower, j_upper;
                i = std::get<0>(*iterator);
                --iterator;
                j = std::get<0>(*iterator);
                j_lower = std::get<1>(*iterator);
                j_upper = std::get<2>(*iterator);
                ++iterator;

                float distance = State::dataset.distance(task.capture_set(), i, j, id);
                std::get<1>(* iterator) = std::max(std::get<1>(* iterator), j_lower - distance);
                std::get<2>(* iterator) = std::min(std::get<2>(* iterator), j_upper + distance);
            }

            { // Comparison with next feature
                unsigned int i, j;
                float j_lower, j_upper;
                i = std::get<0>(*iterator);
                ++iterator;
                if (iterator != bounds -> second.end()) {
                    j = std::get<0>(*iterator);
                    j_lower = std::get<1>(*iterator);
                    j_upper = std::get<2>(*iterator);
                    --iterator;

                    float distance = State::dataset.distance(task.capture_set(), i, j, id);
                    std::get<1>(* iterator) = std::max(std::get<1>(* iterator), j_lower - distance);
                    std::get<2>(* iterator) = std::min(std::get<2>(* iterator), j_upper + distance);
                } else {
                    --iterator;
                }
            }            
        }

        if (std::get<1>(* iterator) > task.upperscope()) { continue; }
        if (std::get<2>(* iterator) < upper) { optimal_feature = std::get<0>(* iterator); }
        lower = std::min(lower, std::get<1>(* iterator));
        upper = std::min(upper, std::get<2>(* iterator));
    }
    return task.update(lower, upper, optimal_feature);
}

bool Optimizer::load_parents(Tile const & identifier, adjacency_accessor & parents) {
    return State::graph.edges.find(parents, identifier);
}

bool Optimizer::load_self(Tile const & identifier, vertex_accessor & self) {
    return State::graph.vertices.find(self, identifier);
}

bool Optimizer::store_self(Tile const & identifier, Task const & value, vertex_accessor & self) {
    return State::graph.vertices.insert(self, std::make_pair(identifier, value));
}

void Optimizer::store_children(Task & task, unsigned int id) {
    bound_accessor bounds;
    bool inserted = State::graph.bounds.insert(bounds, task.identifier());
    if (!inserted) { return; }
    int optimal_feature = -1;
    float lower = task.base_objective(), upper = task.base_objective();
    Bitmask const & features = task.feature_set();
    for (int j_begin = 0, j_end = 0; features.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {

            if (Configuration::feature_transform == false) {
                for (int sign = -1; sign <= 1; sign += 2) {
                    key_type child_key(State::locals[id].neighbourhood[2 * j + (sign < 0 ? 0 : 1)].capture_set(), 0);
                    vertex_accessor child;
                    if (State::graph.vertices.find(child, child_key)) {
                        State::locals[id].neighbourhood[2 * j + (sign < 0 ? 0 : 1)] = child -> second;
                    }
                }
            }

            Task & left = State::locals[id].neighbourhood[2 * j];
            Task & right = State::locals[id].neighbourhood[2 * j + 1];

            float split_lower, split_upper;
            if (Configuration::rule_list) {
                float lower_negative = left.lowerbound() + right.base_objective();
                float lower_positive = left.base_objective() + right.lowerbound();
                split_lower = std::min(lower_negative, lower_positive);
                float upper_negative = left.upperbound() + right.base_objective();
                float upper_positive = left.base_objective() + right.upperbound();
                split_upper = std::min(upper_negative, upper_positive);
            } else {
                split_lower = left.lowerbound() + right.lowerbound();
                split_upper = left.upperbound() + right.upperbound();
            }
            bounds -> second.push_back(std::tuple<int, float, float>(j, split_lower, split_upper));
            if (split_lower > task.upperscope()) { continue; }
            if (split_upper < upper) { optimal_feature = j; }
            lower = std::min(lower, split_lower);
            upper = std::min(upper, split_upper);
        }
    }
    task.update(lower, upper, optimal_feature);
}

void Optimizer::link_to_parent(Tile const & parent, Bitmask const & features, Bitmask const & signs, float scope, Tile const & self, translation_type const & order, adjacency_accessor & parents) {
    for (int j_begin = 0, j_end = 0; features.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            int feature = (signs.get(j) ? 1 : -1) * (j + 1);
            State::graph.translations.insert(std::make_pair(std::make_pair(parent, feature), order)); // insert translation
            State::graph.children.insert(std::make_pair(std::make_pair(parent, feature), self)); // insert forward look-up entry
            State::graph.edges.insert(parents, self); // insert backward look-up entry
            std::pair<adjacency_iterator, bool> insertion = parents -> second.insert(
                std::make_pair(parent, std::make_pair(Bitmask(State::dataset.width(), false), scope)));
            insertion.first -> second.first.set(j, true);
            insertion.first -> second.second = std::min(insertion.first -> second.second, scope);
        }
    }
}

void Optimizer::signal_exploiters(adjacency_accessor & parents, Task & self, unsigned int id) {
    if (self.uncertainty() != 0 && self.lowerbound() < self.lowerscope() - std::numeric_limits<float>::epsilon()) { return; }
    for (adjacency_iterator iterator = parents -> second.begin(); iterator != parents -> second.end(); ++iterator) {
        if (iterator -> second.first.count() == 0) { continue; }
        if (self.lowerbound() < iterator -> second.second - std::numeric_limits<float>::epsilon() && self.uncertainty() > 0) { continue; }
        State::locals[id].outbound_message.exploitation(
            self.identifier(), // sender tile
            iterator -> first, // recipient tile
            iterator -> second.first, // recipient features
            self.support() - self.lowerbound()); // priority
        State::queue.push(State::locals[id].outbound_message);
        // iterator -> second.first.clear(); // reset the dependencies so we don't repeat exploits
    }
}

bool Optimizer::update_root(float lower, float upper) {
    bool change = lower != this -> global_lowerbound || upper != this -> global_upperbound;
    this -> global_lowerbound = lower;
    this -> global_upperbound = upper;
    this -> global_lowerbound = std::min(this -> global_upperbound, this -> global_lowerbound);
    this -> global_boundary = global_upperbound - global_lowerbound;
    return change;
}
