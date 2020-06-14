#include <iostream>
#include <limits>

#include "task.hpp"

Task::Task(void) {}

Task::Task(float const lowerbound, float const upperbound, float const support, float base_objective)
 : _support(support),
  _lowerbound(lowerbound), _upperbound(upperbound),
  _potential(upperbound - lowerbound), _base_objective(base_objective) {
    float const epsilon = std::numeric_limits<float>::epsilon();    
    if (lowerbound - upperbound > epsilon) {
        std::cout << "Invalid Task(" << lowerbound << ", " << upperbound << ")" << std::endl;
        throw "Invalid Task Initialization";
    }
}

Task::Task(float const lowerbound, float const upperbound, float const support, float base_objective, Bitmask const & sensitivity, similarity_index_table_type const & index)
 : _support(support),
  _lowerbound(lowerbound), _upperbound(upperbound),
  _potential(upperbound - lowerbound), _base_objective(base_objective),
  _sensitivity(sensitivity), similarity_index(index) {
    float const epsilon = std::numeric_limits<float>::epsilon();    
    if (lowerbound - upperbound > epsilon) {
        std::cout << "Invalid Task(" << lowerbound << ", " << upperbound << ")" << std::endl;
        throw "Invalid Task Initialization";
    }
}

float const Task::support(void) const {
    return this -> _support;
}

float const Task::lowerbound(void) const {
    return this -> _lowerbound;
}

float const Task::upperbound(void) const {
    return this -> _upperbound;
}

float const Task::potential(void) const {
    return this -> _potential;
}

float const Task::uncertainty(void) const {
    float const epsilon = std::numeric_limits<float>::epsilon();
    if (upperbound() - lowerbound() > epsilon) {
        return upperbound() - lowerbound();
    } else {
        return 0.0;
    }
}

float const Task::objective(void) const {
    if (uncertainty() != 0) { throw "Cannot determine objective from task with non-zero uncertainty."; }
    return upperbound();
}

float const Task::base_objective(void) const {
    return this -> _base_objective;
}

float const Task::scope(void) const {
    return this -> _scope;
}

void Task::rescope(float scope_value) {
    this -> _scope = scope_value;
}

Bitmask const & Task::sensitivity(void) const {
    return this -> _sensitivity;
}
bool const Task::sensitive(int const index) const {
    return this -> _sensitivity[index] == 1;
}
void Task::desensitize(int const index) {
    this -> _sensitivity.set(index, false);
    return;
}

float const Task::priority(float const optimism) const {
    return 1.0 - ( optimism * lowerbound() + (1 - optimism) * upperbound() ) / support();
}


void Task::inform(float const proposed_lowerbound, float const proposed_upperbound) {
    float const epsilon = std::numeric_limits<float>::epsilon();
    float lowerbound = std::max(this -> _lowerbound, proposed_lowerbound);
    float upperbound = std::min(this -> _upperbound, proposed_upperbound);
    if (lowerbound - upperbound > epsilon) {
        // std::cout << "Invalid Task::inform (" << this -> _lowerbound << ", " << this -> _upperbound << ") => (" << lowerbound << ", " << upperbound << ")" << std::endl;
        throw "Invalid Lower or Upper Bound Update to Task Instance.";
    }
    this -> _lowerbound = lowerbound;
    this -> _upperbound = upperbound;
    return;
}

bool const Task::explored(void) const {
    return this -> _explored;
}
bool const Task::delegated(void) const {
    return this -> _delegated;
}
bool const Task::cancelled(void) const {
    return this -> _cancelled;
}
bool const Task::resolved(void) const {
    return this -> _resolved;
}

void Task::explore(void) {
    if (cancelled() || resolved()) {
        std::cout << "Called explore on task.cancelled == " << cancelled() << ", task.resolved == " << resolved() << std::endl;
        throw "Illegal task state transition.";
    }
    this -> _explored = true;
    return;
}
void Task::delegate(void) {
    if (cancelled() || resolved()) {
        std::cout << "Called delegate on task.cancelled == " << cancelled() << ", task.resolved == " << resolved() << std::endl;
        throw "Illegal task state transition.";
    }
    this -> _delegated = true;
    return;
}
void Task::cancel(void) {
    if (resolved()) {
        std::cout << "Called cancel on task.cancelled == " << cancelled() << ", task.resolved == " << resolved() << std::endl;
        throw "Illegal task state transition.";
    }
    this -> _cancelled = true;
    return;
}
void Task::resolve(void) {
    if (cancelled()) {
        std::cout << "Called resolve on task.cancelled == " << cancelled() << ", task.resolved == " << resolved() << std::endl;
        throw "Illegal task state transition.";
    }
    this -> _resolved = true;
    this -> _lowerbound = this -> _upperbound;
    return;
}
