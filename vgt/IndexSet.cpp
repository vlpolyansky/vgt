#include "IndexSet.h"
#include <algorithm>
#include <iostream>

int IndexSet::hash() const {
    if (!hash_initialized) {
        size_t seed = size();
        for (size_t i = 0; i < size(); i++) {
            seed ^= this->operator[](i) + 0x9e3779b9 + (seed << 6u) + (seed >> 2u);
        }
        hash_value = seed;
        hash_initialized = true;
    }
    return hash_value;
}

IndexSet::IndexSet(const vec<int> &index_set) : vec<int>(index_set) {
    // assert(!index_set.empty());
}

IndexSet::IndexSet(std::initializer_list<int> indices) : vec<int>(indices) {
    // assert(indices.size() > 0);
}

bool IndexSet::operator==(const IndexSet &rhs) const {
    const vec<int> &self = *this;
    return hash() == rhs.hash() && self == rhs;
}

bool IndexSet::operator!=(const IndexSet &rhs) const {
    return !(rhs == *this);
}

int IndexSet::dim() const {
    // assert(!empty());
    return int(size()) - 1;
}

IndexSet IndexSet::append(int p) const {
    vec<int> result(this->begin(), this->end());
    result.insert(std::upper_bound(result.begin(), result.end(), p), p);
    return result;
}

IndexSet IndexSet::remove_at(int idx) const {
    assert(idx < size());
    vec<int> vi(begin(), begin() + idx);
    vi.insert(vi.end(), begin() + idx + 1, end());
    return vi;
}

bool IndexSet::contains(int p) const {
    return std::find(begin(), end(), p) != end();
}

std::ostream &operator<<(std::ostream &os, const IndexSet &set) {
    os << set.dim();
    for (size_t i = 0; i < set.size(); i++) {
        os << " " << set[i];
    }
    return os;
}

std::size_t std::hash<IndexSet>::operator()(const IndexSet &k) const {
    return k.hash();
}

vec<IndexSet> IndexSet::boundary() const {
    vec<IndexSet> b;
    if (dim() == 0) {
        return b;
    }
    for (size_t i = 0; i < size(); i++) {
        b.push_back(remove_at(i));
    }
    return b;
}
