#include "state.hpp"

Dataset State::dataset = Dataset();
Graph State::graph = Graph();
Queue State::queue = Queue();
std::vector< LocalState > State::locals = std::vector< LocalState >();
int State::status = 0;

void State::initialize(std::istream & data_source, unsigned int workers) {
    State::dataset.load(data_source);
    State::graph = Graph();
    State::queue = Queue();
    State::locals.resize(workers);
    for (unsigned int i = 0; i < workers; ++i) {
        State::locals[i].initialize(dataset.height(), dataset.width(), dataset.depth());
    }
}

void State::reset(void) {
    State::graph = Graph();
    State::queue = Queue();
    State::locals.clear();
    State::dataset.clear();
}