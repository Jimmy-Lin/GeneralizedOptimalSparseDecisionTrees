void Optimizer::diagnose_false_convergence(void) {
    // diagnose_false_convergence(this -> root_set);
    return;
}
bool Optimizer::diagnose_false_convergence(key_type const & key) {
    if (Configuration::diagnostics == false) { return false; }
    std::unordered_set< Model * > results;
    models(key, results);
    if (results.size() > 0) { return false; }

    // unsigned int m = State::dataset.width();

    // float epsilon = std::numeric_limits<float>::epsilon();
    vertex_accessor task;
    State::graph.vertices.find(task, key);

    std::cout
        << "Task(" << task -> second.capture_set().to_string() << ") is falsely convergent."
        << " Bounds = " << "[" << task -> second.lowerbound() << ", " << task -> second.upperbound() << "]"
        << ", Base = " << task -> second.base_objective() << std::endl;
    
    bound_accessor bounds;
    State::graph.bounds.find(bounds, task -> second.identifier());
    for (bound_iterator iterator = bounds -> second.begin(); iterator != bounds -> second.end(); ++iterator) {
        int feature = std::get<0>(* iterator);
        bool ready;
        float lower = 0.0, upper = 0.0;
        for (int sign = -1; sign <= 1; sign += 2) {
            vertex_accessor child;
            child_accessor key;
            ready = ready && State::graph.children.find(key, std::make_pair(task -> second.identifier(), sign * (feature + 1)))
                && State::graph.vertices.find(child, key -> second);
            if (ready) {
                lower += child -> second.lowerbound();
                upper += child -> second.upperbound();
            }
        }
        if (ready) {
            std::get<1>(* iterator) = lower;
            std::get<2>(* iterator) = upper;
        }

        if (std::get<2>(* iterator) > task -> second.upperbound() + std::numeric_limits<float>::epsilon()) { continue; }

        std::cout << "Task(" << key.to_string() << ")'s upper bound points to Feature " << feature << std::endl;

        {
            vertex_accessor child;
            child_accessor key;
            if (State::graph.children.find(key, std::make_pair(task -> second.identifier(), (feature + 1)))) {
                diagnose_false_convergence(key -> second);
            }
        }
        {
            vertex_accessor child;
            child_accessor key;
            if (State::graph.children.find(key, std::make_pair(task -> second.identifier(), -(feature + 1)))) {
                diagnose_false_convergence(key -> second);
            }
        }
    }
    return false;
}