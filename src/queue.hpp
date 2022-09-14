#ifndef QUEUE_H
#define QUEUE_H

#include <iostream>
#include <tuple>
#include <unordered_set>

#include <tbb/concurrent_priority_queue.h>
#include <tbb/concurrent_hash_map.h>
#include <tbb/concurrent_vector.h>
#include <tbb/scalable_allocator.h>

#include "bitmask.hpp"
#include "configuration.hpp"
#include "message.hpp"

typedef Message message_type;

class PriorityKeyComparator {
public:
    // Note that the tbb::concurrent_priority_queue is implemented to pop the item with the highest priority value
    bool operator()(message_type const * left, message_type const * right) {
        return (* left) < (* right);
    }
};

struct MembershipKeyHashCompare {
    static size_t hash(message_type * message) {
        return message -> hash();
    }
    static bool equal(message_type * left, message_type * right) {
        if ((* left) == (* right)) {
            left -> features.bit_or(right -> features);
            right -> features.bit_or(left -> features);
            left -> signs.bit_or(right -> signs);
            right -> signs.bit_or(left -> signs);
            left -> scope = std::max(left -> scope, right -> scope);
            right -> scope = std::max(left -> scope, right -> scope);
            return true;
        } else {
            return false;
        }
    }
};

typedef tbb::concurrent_priority_queue< message_type *, PriorityKeyComparator, 
    tbb::scalable_allocator< message_type * > > queue_type;

typedef tbb::concurrent_hash_map< message_type *, bool, MembershipKeyHashCompare, 
    tbb::scalable_allocator<std::pair<message_type * const, bool>>> membership_table_type; // FIREWOLF: Fix the static assertion error

class Queue {
public:
    Queue(void);
    ~Queue(void);

    // @param message: a message to be sent from one vertex to another
    // @returns true if the message was successfully enqueued and not rejected by the membership filter
    // @note higher priority comes before lower priority
    bool push(Message const & message);
    
    // @returns whether queue is empty
    bool empty(void) const;

    // @returns the size of the queue
    unsigned int size(void) const;

    // @requires message: the 4th item of message must contain an instance of identifier_type with the correct amount of pre-allocated memory to copy assign into
    // @param message: a tuple containing the bitmask address, blocks
    // @param index: the particular channel (one of the queues) to pop from
    // @modifes message: message will be overwritten with a copy the content of the received message
    bool pop(Message & message);

private:
    // map containing uniquely identified messages that are currently in queue
    membership_table_type membership;

    queue_type queue; // queue containing pending messages
};

#endif