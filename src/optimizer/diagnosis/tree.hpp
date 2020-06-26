
void Optimizer::diagnostic_tree(int iteration) {
    json tracer = json::object();
    tracer["directed"] = true;
    tracer["multigraph"] = false;
    tracer["graph"] = json::object();
    tracer["graph"]["name"] = "GOSDT Trace";
    tracer["links"] = json::array();
    tracer["nodes"] = json::array();
    diagnostic_tree(this -> root, tracer);

    int indentation = 2;

    std::stringstream trace_name;
    trace_name << Configuration::tree << "/" << iteration << ".gml";
    std::string trace_result = tracer.dump(indentation);
    std::ofstream out(trace_name.str());
    out << trace_result;
    out.close();

    return;
}
bool Optimizer::diagnostic_tree(key_type const & identifier, json & tracer) {
    vertex_accessor task_accessor;
    if (State::graph.vertices.find(task_accessor, identifier) == false) { return false; }
    Task & task = task_accessor -> second;

    json node = json::object();
    node["id"] = identifier.to_string();
    node["capture"] = task.capture_set().to_string();
    node["support"] = task.support();
    node["terminal"] = task.lowerbound() == task.upperbound();

    
    if (task.lowerbound() == task.base_objective()) { 
        tracer["nodes"].push_back(node);
        return true;
    }

    json scores = json::object();

    unsigned int m = State::dataset.width();
    unsigned int k = 0;
    float score_k = std::numeric_limits<float>::max();

    bound_accessor bounds;
    State::graph.bounds.find(bounds, task.identifier());
    for (bound_iterator iterator = bounds -> second.begin(); iterator != bounds -> second.end(); ++iterator) {
        int feature = std::get<0>(* iterator);

        std::string type, relation, reference;
        State::dataset.encoder.encoding(feature, type, relation, reference);
        float upper = std::get<2>(* iterator);
        scores[reference] = upper;
        if (upper < score_k) {
            score_k = upper;
            k = feature;
        }
    }
    unsigned int decoded_index;
    std::string type, relation, reference;
    State::dataset.encoder.decode(k, & decoded_index);
    State::dataset.encoder.encoding(k, type, relation, reference);
    node["threshold"] = reference;
    node["scores"] = scores;
    tracer["nodes"].push_back(node);
    if (score_k < std::numeric_limits<float>::max()) {
        child_accessor left_key, right_key;
        if (State::graph.children.find(left_key, std::make_pair(identifier, -(k + 1)))) {    
            diagnostic_tree(left_key -> second, tracer);
            left_key.release();
        }
        if (State::graph.children.find(right_key, std::make_pair(identifier, k + 1))) {
            diagnostic_tree(right_key -> second, tracer);
            right_key.release();
        }
    }

    return true;
}
