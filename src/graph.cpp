#include "graph.hpp"

Graph::Graph(void) {}

Graph::~Graph(void) {}

// bool Graph::exists(key_type const & key) const {
//     return this -> vertices.count(key) != 0;
// }

// bool Graph::insert(key_type const & key, value_type const & value) {
//     bool hit = this -> vertices.insert(std::make_pair(key, value));
//     // this -> forward_index.insert(std::make_pair(key, index_type()));
//     // this -> backward_index.insert(std::make_pair(key, index_type()));
//     this -> backward_index.insert(std::pair< key_type, index_type >(
//         std::piecewise_construct, std::forward_as_tuple(key), std::forward_as_tuple()));
//     return hit;
// }

// bool Graph::insert(std::pair< key_type, value_type > const & pair) {
//     bool hit = this -> vertices.insert(pair);
//     // this -> forward_index.insert(std::make_pair(key, index_type()));
//     this -> backward_index.insert(std::pair< key_type, index_type >(
//         std::piecewise_construct, std::forward_as_tuple(pair.first), std::forward_as_tuple()));
//     return hit;
// }

// bool Graph::connect(key_type const & parent, key_type const & child, float scope) {
//     // { // insert forward edge
//     //         if (this -> forward_index.count(parent) == 0) {
//     //             this -> forward_index.insert(std::make_pair(parent, index_type()));
//     //         }
//     //     index_table::accessor accessor;
//     //     if (find(accessor, parent, true) == false) { return false; };
//     //     accessor -> second[child] = feature_index;
//     // }
//     // { // insert backward edge
//     //     if (this -> backward_index.count(child) == 0) {
//     //         this -> backward_index.insert(std::make_pair(child, index_type()));
//     //     }
//     //     index_table::accessor accessor;
//     //     if (find(accessor, child, false) == false) { return false; };
//     //     std::pair< index_type::iterator, bool > insertion = accessor -> second.emplace_back(parent, scope);
//     //     if (insertion.second == true) {
//     //         this -> edge_counter += 1;
//     //     };
//     // }
//     return true;
// }

// bool Graph::find(const_vertex_accessor & accessor, key_type const & key) const {
//     return this -> vertices.find(accessor, key);
// }

// bool Graph::find(vertex_accessor & accessor, key_type const & key) const {
//     return this -> vertices.find(accessor, key);
// }

// bool Graph::find(const_index_accessor & accessor, key_type const & key, bool forward) const {
//     if (forward) {
//         return this -> forward_index.find(accessor, key);
//     }     else {
//         return this -> backward_index.find(accessor, key);
//     }
// }

// bool Graph::find(index_accessor & accessor, key_type const & key, bool forward) const {
//     if (forward) {
//         return this -> forward_index.find(accessor, key);
//     } else {
//         return this -> backward_index.find(accessor, key);
//     }
// }

// // bool Graph::find_or_create(const_vertex_accessor & accessor, key_type const & key,
// //     Bitmask & buffer_1, Bitmask & buffer_2, Bitmask & buffer_3,
// //     Task const & task, unsigned int index, bool condition) {

// //     Bitmask sensitivity(task.sensitivity());
// //     sensitivity.set(index, false);

// //     buffer_1 = key;
// //     this -> dataset.subset(index, condition, buffer_1); // Compute the subset

// //     // if (exists(buffer_1) == false) {
// //     vertex_type vertex(
// //         std::piecewise_construct,
// //         std::forward_as_tuple(buffer_1),
// //         std::forward_as_tuple(buffer_1, buffer_2, buffer_3, sensitivity, dataset, this -> regularization)
// //     );
// //     bool hit = this -> vertices.insert(accessor, vertex) == false;
// //     if (hit) { this -> hits += 1; }
// //     // } else { this -> hits += 1; }

// //     buffer_1 = key;
// //     this -> dataset.subset(index, condition, buffer_1); // Re-compute the subset (possibly modified by new_task)

// //     if (connect(key, buffer_1, index) == false) {
// //         std::stringstream reason;
// //         reason << "Failed to find indices for vertex (" << buffer_1.to_string() << "), child to vertex (" << key.to_string() << ") for creating connection.";
// //         throw IntegrityViolation("Optimizer::execute", reason.str());
// //     }

// //     return true;
// // }

// // bool Graph::find_or_create(vertex_accessor & accessor, key_type const & key,
// //     Bitmask & buffer_1, Bitmask & buffer_2, Bitmask & buffer_3,
// //     Task const & task, unsigned int index, bool condition) {

// //     Bitmask sensitivity(task.sensitivity());
// //     sensitivity.set(index, false);

// //     buffer_1 = key;
// //     this -> dataset.subset(index, condition, buffer_1); // Compute the subset

// //     // if (exists(buffer_1) == false) {
// //     vertex_type vertex(
// //         std::piecewise_construct,
// //         std::forward_as_tuple(buffer_1),
// //         std::forward_as_tuple(buffer_1, buffer_2, buffer_3, sensitivity, dataset, this -> regularization)
// //     );
// //     bool hit = this -> vertices.insert(accessor, vertex) == false;
// //     if (hit) { this -> hits += 1; }
// //     // } else { this -> hits += 1; }

// //     buffer_1 = key;
// //     this -> dataset.subset(index, condition, buffer_1); // Re-compute the subset (possibly modified by new_task)

// //     if (connect(key, buffer_1, index) == false) {
// //         std::stringstream reason;
// //         reason << "Failed to find indices for vertex (" << buffer_1.to_string() << "), child to vertex (" << key.to_string() << ") for creating connection.";
// //         throw IntegrityViolation("Optimizer::execute", reason.str());
// //     }

// //     return true;
// // }

// bool Graph::erase(key_type const & key, bool disconnect) {
//     if (this -> vertices.erase(key)) {
//         if (disconnect) {
//             // { // Handle forward index
//             //     index_accessor outer;
//             //     find(outer, key, true);
//             //     for (auto it = outer -> second.begin(); it != outer -> second.end(); ++it) {
//             //         index_accessor inner;
//             //         find(inner, it -> first, false);
//             //         inner -> second.erase(key);
//             //     }
//             // }
//             // { // Handle backward index
//             //     index_accessor outer;
//             //     find(outer, key, false);
//             //     for (auto it = outer -> second.begin(); it != outer -> second.end(); ++it) {
//             //         index_accessor inner;
//             //         find(inner, it -> first, true);
//             //         inner -> second.unsafe_erase(key);
//             //     }
//             // }
//         }
//         this -> forward_index.erase(key);
//         this -> backward_index.erase(key);
//         return true;
//     } else {
//         return false;
//     }
// }

// bool Graph::disconnect(key_type const & parent, key_type const & child) {
//     if (!exists(parent) || !exists(child)) { return false; }
    
//     // { // remove forward edge
//     //     index_table::accessor accessor;
//     //     if (find(accessor, parent, true) == false) { return false; };
//     //     accessor -> second.erase(child);
//     // }
//     { // remove backward edge
//         index_table::accessor accessor;
//         if (find(accessor, child, false) == false) { return false; };

//         // accessor -> second.unsafe_erase(parent);
        
//         // connection_iterator iterator = accessor -> second.find(parent);
//         // if (iterator == accessor -> second.end()) { return false; }
//         // iterator -> second = -1; // Mark connection as disabled

//         this -> edge_counter -= 1;
//     }
//     return true;
// }

void Graph::clear(void) {
    this -> vertices.clear();
    this -> edges.clear();
    this -> translations.clear();
    this -> children.clear();
    this -> vertices.clear();
    this -> bounds.clear();
    return;
}

unsigned int Graph::size(void) const {
    return this -> vertices.size();
}