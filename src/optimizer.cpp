#include "optimizer.hpp"

#include "optimizer/diagnosis/false_convergence.hpp"
#include "optimizer/diagnosis/non_convergence.hpp"
#include "optimizer/diagnosis/trace.hpp"
#include "optimizer/diagnosis/tree.hpp"
#include "optimizer/dispatch/dispatch.hpp"
#include "optimizer/extraction/models.hpp"

Optimizer::Optimizer(void) {
    return;
}

Optimizer::~Optimizer(void) {
    State::reset();
    return;
}



void Optimizer::load(std::istream & data_source) { State::initialize(data_source, Configuration::worker_limit); }

void Optimizer::reset(void) { State::reset(); }

void Optimizer::initialize(void) {
    // Initialize Profile Output
    if (Configuration::profile != "") {
        std::ofstream profile_output(Configuration::profile);
        profile_output << "iterations,time,lowerbound,upperbound,graph_size,queue_size,explore,exploit";
        profile_output << std::endl;
        profile_output.flush();
    }

    // Initialize Timing State
    this -> start_time = tbb::tick_count::now();

    int const n = State::dataset.height();
    int const m = State::dataset.width();

    // Enqueue for exploration
    State::locals[0].outbound_message.exploration(Tile(), Bitmask(n, true, NULL, Configuration::depth_budget), Bitmask(m, true), 0, std::numeric_limits<float>::max());
    State::queue.push(State::locals[0].outbound_message);
    return;
}


void Optimizer::objective_boundary(float * lowerbound, float * upperbound) const {
    * lowerbound = this -> global_lowerbound;
    * upperbound = this -> global_upperbound;
}

float Optimizer::uncertainty(void) const {
    float const epsilon = std::numeric_limits<float>::epsilon();
    float value = this -> global_upperbound - this -> global_lowerbound;
    return value < epsilon ? 0 : value;
}

float Optimizer::elapsed(void) const {
    auto now = tbb::tick_count::now();
    float duration = (now - this -> start_time).seconds();
    return duration;
}

bool Optimizer::timeout(void) const {
    return (Configuration::time_limit > 0 && elapsed() > Configuration::time_limit);
}

bool Optimizer::complete(void) const {
    return uncertainty() == 0;
}

unsigned int Optimizer::size(void) const {
    return State::graph.size();
}

bool Optimizer::iterate(unsigned int id) {
    bool update = false;
    if (State::queue.pop(State::locals[id].inbound_message)) {
        update = dispatch(State::locals[id].inbound_message, id);
        switch (State::locals[id].inbound_message.code) {
            case Message::exploration_message: { this -> explore += 1; break; }
            case Message::exploitation_message: { this -> exploit += 1; break; }
        }
    }

    // Worker 0 is responsible for managing ticks and snapshots
    if (id == 0) {
        this -> ticks += 1;

        // snapshots that would need to occur every iteration
        // if (Configuration::trace != "") { this -> diagnostic_trace(this -> ticks, State::locals[id].message); }
        if (Configuration::tree != "") { this -> diagnostic_tree(this -> ticks); }

        // snapshots that can skip unimportant iterations
        if (update || complete() || ((this -> ticks) % (this -> tick_duration)) == 0) { // Periodic check for completion for timeout
            // Update the continuation flag for all threads
            this -> active = !complete() && !timeout() && (Configuration::worker_limit > 1 || State::queue.size() > 0);
            this -> print();
            this -> profile();
        }
    }
    return this -> active;
}

void Optimizer::print(void) const {
    if (Configuration::verbose) { // print progress to standard output
        float lowerbound, upperbound;
        objective_boundary(& lowerbound, & upperbound);
        std::cout <<
            "Time: " << elapsed() <<
            ", Objective: [" << lowerbound << ", " << upperbound << "]" <<
            ", Boundary: " << this -> global_boundary <<
            ", Graph Size: " << State::graph.size() <<
            ", Queue Size: " << State::queue.size() << std::endl;
    }
}

void Optimizer::profile(void) {
    if (Configuration::profile != "") {
        std::ofstream profile_output(Configuration::profile, std::ios_base::app);
        float lowerbound, upperbound;
        objective_boundary(& lowerbound, & upperbound);
        profile_output << this -> ticks << "," << elapsed() << "," <<
            lowerbound << "," << upperbound << "," << State::graph.size() << "," << 
            State::queue.size() << "," << this -> explore << "," << this -> exploit;
        profile_output << std::endl;
        profile_output.flush();
        this -> explore = 0;
        this -> exploit = 0;
    }
}

float Optimizer::cart(Bitmask const & capture_set, Bitmask const & feature_set, unsigned int id) const {
    Bitmask left(State::dataset.height());
    Bitmask right(State::dataset.height());
    float potential, min_loss, max_loss, base_info;
    unsigned int target_index;
    State::dataset.summary(capture_set, base_info, potential, min_loss, max_loss, target_index, id);
    float base_risk = max_loss + Configuration::regularization;

    if (max_loss - min_loss < Configuration::regularization
        || 1.0 - min_loss < Configuration::regularization
        || (potential < 2 * Configuration::regularization && (1.0 - max_loss) < Configuration::regularization)
        || feature_set.empty()) {
        return base_risk;
    }

    int information_maximizer = -1;
    float information_gain = 0;
    for (int j_begin = 0, j_end = 0; feature_set.scan_range(true, j_begin, j_end); j_begin = j_end) {
        for (int j = j_begin; j < j_end; ++j) {
            float left_info, right_info;
            left = capture_set;
            right = capture_set;
            State::dataset.subset(j, false, left);
            State::dataset.subset(j, true, right);

            if (left.empty() || right.empty()) { continue; }

            State::dataset.summary(capture_set, left_info, potential, min_loss, max_loss, target_index, id);
            State::dataset.summary(capture_set, right_info, potential, min_loss, max_loss, target_index, id);

            float gain = left_info + right_info - base_info;
            if (gain > information_gain) {
                information_maximizer = j;
                information_gain = gain;
            }
        }
    }

    if (information_maximizer == -1) { return base_risk; }

    left = capture_set;
    right = capture_set;
    State::dataset.subset(information_maximizer, false, left);
    State::dataset.subset(information_maximizer, true, right);
    float risk = cart(left, feature_set, id) + cart(right, feature_set, id);
    return std::min(risk, base_risk);
}
