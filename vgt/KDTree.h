#pragma once

#include "utils.h"
#include "IndexSet.h"

class KDTree {
private:
    class Node;
//    typedef std::shared_ptr<Node> pNode;
    typedef int pNode;

public:
    explicit KDTree(const dmatrix &data, int leaf_len = 5);

    void init();

    int find_nn(const dvector &x, ftype *best_dist_sqr, ftype margin_sqr = -1,
                const vec<int> &ignore = vec<int>()) const;

private:
    pNode build_tree(int l, int r, int depth);
    void find_nn(const dvector &x, pNode node, int *best, ftype *best_dist_sqr,
                 ftype margin_sqr, ftype cur_bin_dist_sqr, const vec<int> &ignore,
                 vec<ftype> &partial_dist_sqr) const;

    pNode make_node(int lidx, int ridx);

    pNode make_node(int dim, ftype m);

private:
    const dmatrix &data;
    int n;
    int d;
    pNode head;
    int leaf_len;

    vec<int> indices;
    vec<Node> nodes;

    struct Node {
        Node(int lidx, int ridx);
        Node(int dim, ftype m);

        int lidx, ridx;

        int dim;
        ftype m;

        pNode left = -1, right = -1;
    };
};

