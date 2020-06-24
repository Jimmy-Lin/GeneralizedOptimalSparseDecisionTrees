#include "queue.hpp"

Queue::Queue(void) {
    return;
}

Queue::~Queue(void) {
    return;
}

void Queue::initialize(int width, int height, float equity) {
    this -> _width = width;
    this -> _height = height;
    this -> _equity = equity;
    
    for (int j = 0; j < width;  ++j) {
        std::vector< float > identity;
        identity.resize(height, 0.5); // Create a new unbiased identity vector
        this -> identities.emplace_back(identity, 1);
        this -> queues.emplace_back();
        // this -> queues.push_back(tbb::concurrent_priority_queue< std::tuple< Key, float, float, float >, PriorityKeyComparator >); // Create a new queue
    }
}

void Queue::push(Key const & key, float const primary_priority, float const secondary_priority, float const tertiary_priority) {
    if (this -> membership.insert(std::make_pair(key, true))) {
        // Decide which queue is the most fitting
        int minimum_index = 0;
        float minimum_distance = 1.0;
        float equity = this -> _equity;
        float diversity = 1.0;

        for (int j = 0; j < width(); ++j) {
            float difference_value = difference(j, key.indicator());
            float saturation_value = saturation(j);
            diversity = std::min(difference_value, diversity);
            float distance = (1 - equity) * difference_value + equity * saturation_value;
            if (distance < minimum_distance) {
                minimum_distance = distance;
                minimum_index = j;
            }
        }
        assimilate(minimum_index, key.indicator());
        this -> queues[minimum_index].push(queue_element(key, primary_priority, secondary_priority, tertiary_priority + diversity));
    }
    return;
}



float Queue::difference(int const index, Bitmask const & instance) const {
    std::vector< float > const & group_identity = std::get<0>(this -> identities[index]);
    float difference = 0;
    for (int i = 0; i < height(); ++i) {
        difference += std::fabs((float)(group_identity[i]) - (float) (instance[i])) / (float) height();
    }
    return difference;
}
float Queue::saturation(int const index) const {
    int minimum_size = this -> queues[0].size();
    int maximum_size = this -> queues[0].size();
    int reference_size = this -> queues[index].size();

    for (int j = 0; j < width(); ++j) {
        minimum_size = std::min(minimum_size, (int) (this -> queues[j].size()));
        maximum_size = std::max(maximum_size, (int) (this -> queues[j].size()));
    }
    float saturation = (float) std::abs(reference_size - minimum_size); // / (float) (maximum_size - minimum_size);
    return saturation;
}



void Queue::assimilate(int const index, Bitmask const & instance) {
    std::tuple< std::vector< float >, int > & group_identity = this -> identities[index]; // mutable reference
    std::vector< float > & group_identity_vector = std::get<0>(group_identity); // mutable reference
    float sample_size = std::get<1>(group_identity);

    for (int i = 0; i < height(); ++i) {
        group_identity_vector[i] = (group_identity_vector[i] * sample_size + instance[i]) / (sample_size + 1);
    }
    std::get<1>(group_identity) += 1;
    return;
}


bool const Queue::empty(void) const {
    return size() == 0;
}

int const Queue::size(void) const {
    int size = 0;
    for (int j = 0; j < width(); ++j) {
        size += this -> queues[j].size();
    }
    return size;
}

int const Queue::local_size(int const index) const {
    return this -> queues[index].size();
}

bool Queue::pop(std::tuple< Key, float, float, float > & item, int const index) {
    if (this -> queues[index % width()].try_pop(item)) {
        Key key = std::get< 0 >(item);
        this -> membership.erase(key);
        return true;
    } else {
        return false;
    }
}

int const Queue::width(void) const {
    return this -> _width;
}
int const Queue::height(void) const {
    return this -> _height;
}
