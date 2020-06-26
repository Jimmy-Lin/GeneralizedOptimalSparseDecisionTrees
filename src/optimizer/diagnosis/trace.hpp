
void Optimizer::diagnostic_trace(int iteration, key_type const & focal_point) {
    json tracer = json::object();
    tracer["directed"] = true;
    tracer["multigraph"] = false;
    tracer["graph"] = json::object();
    tracer["graph"]["name"] = "GOSDT Trace";
    tracer["links"] = json::array();
    tracer["nodes"] = json::array();
    diagnostic_trace(this -> root, tracer, focal_point);

    int indentation = 2;

    std::stringstream trace_name;
    trace_name << Configuration::trace << "/" << iteration << ".gml";
    std::string trace_result = tracer.dump(indentation);
    std::ofstream out(trace_name.str());
    out << trace_result;
    out.close();
    return;
}
bool Optimizer::diagnostic_trace(key_type const & identifier, json & tracer, key_type const & focal_point) {
    vertex_accessor task_accessor;
    if (State::graph.vertices.find(task_accessor, identifier) == false) { return false; }
    Task & task = task_accessor -> second;

    json node = json::object();
    node["id"] = task.identifier().to_string();
    node["name"] = task.capture_set().to_string();
    node["lowerbound"] = task.lowerbound();
    node["upperbound"] = task.upperbound();
    node["explored"] = true; // new version no longer stores unexplored nodes
    node["resolved"] = task.lowerbound() == task.upperbound();
    node["focused"] = (task.identifier() == focal_point);
    tracer["nodes"].push_back(node);

    bound_accessor bounds;
    State::graph.bounds.find(bounds, task.identifier());
    for (bound_iterator iterator = bounds -> second.begin(); iterator != bounds -> second.end(); ++iterator) {
        int feature = std::get<0>(* iterator);

        child_accessor left_key, right_key;
        if (State::graph.children.find(left_key, std::make_pair(identifier, -(feature + 1)))) {
            json link = json::object();
            link["source"] = identifier.to_string();
            link["target"] = left_key -> second.to_string();
            link["feature"] = feature;
            link["condition"] = false;
            tracer["links"].push_back(link);
            diagnostic_trace(left_key -> second, tracer, focal_point);
            left_key.release();
        }
        if (State::graph.children.find(right_key, std::make_pair(identifier, feature + 1))) {
            json link = json::object();
            link["source"] = identifier.to_string();
            link["target"] = right_key -> second.to_string();
            link["feature"] = feature;
            link["condition"] = true;
            tracer["links"].push_back(link);
            diagnostic_trace(right_key -> second, tracer, focal_point);
            right_key.release();
        }
    }
    return true;
}
