#include "queue.hpp"

Queue::Queue(void) {
    return;
}

Queue::~Queue(void) {
    return;
}

bool Queue::push(Message const & message) {
    message_type * internal_message = new message_type();
    * internal_message = message;

    // Attempt to copy content into membership set
    if (this -> membership.insert(std::make_pair(internal_message, true))) {
        this -> queue.push(internal_message);
        return true;
    } else {
        delete internal_message;
        return false;
    }
}

bool Queue::empty(void) const { return size() == 0; }

unsigned int Queue::size(void) const { return this -> queue.size(); }


bool Queue::pop(Message & message) {
    message_type * internal_message;
    if (this -> queue.try_pop(internal_message)) {
        this -> membership.erase(internal_message); // remove membership
        message = * internal_message;

        delete internal_message;
        return true;
    } else {
        return false;
    }
}
