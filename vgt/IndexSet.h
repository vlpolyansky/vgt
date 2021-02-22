#pragma once

#include <ostream>
#include "utils.h"

class IndexSet : public vec<int> {
public:
    IndexSet(const vec<int> &v);

    IndexSet(std::initializer_list<int> init);

    int hash() const;

    bool operator==(const IndexSet &rhs) const;

    bool operator!=(const IndexSet &rhs) const;

    int dim() const;

    IndexSet append(int p) const;

    IndexSet remove_at(int idx) const;

    bool contains(int p) const;

    friend std::ostream &operator<<(std::ostream &os, const IndexSet &set);

    vec<IndexSet> boundary() const;

private:
    mutable bool hash_initialized = false;
    mutable int hash_value = 0;
};

namespace std {
    template <>
    struct hash<IndexSet> {
        std::size_t operator()(const IndexSet& k) const;
    };
}