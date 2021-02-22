#include <algorithm>
#include "KDTree.h"

KDTree::KDTree(const dmatrix &data, int leaf_len) : data(data), leaf_len(leaf_len),
                                                    n(data.cols()), d(data.rows()), head(-1) {
}

void KDTree::init() {
    for (int i = 0; i < data.cols(); i++) {
        indices.push_back(i);
    }
    head = build_tree(0, int(data.cols()) - 1, 0);
}

KDTree::pNode KDTree::make_node(int lidx, int ridx) {
    nodes.emplace_back(lidx, ridx);
    return static_cast<int>(nodes.size()) - 1;
}

KDTree::pNode KDTree::make_node(int dim, ftype m) {
    nodes.emplace_back(dim, m);
    return static_cast<int>(nodes.size()) - 1;
}

KDTree::pNode KDTree::build_tree(int l, int r, int depth) {
    if (l > r) {
        return -1;
    }
    if (r - l + 1 <= leaf_len) {
        return make_node(l, r);
    }
    int best_dim = -1;
    ftype best_spread = -1;
    for (int i = 0; i < d; i++) {
        ftype mn = data(i, l), mx = data(i, l);
        for (int j = l + 1; j <= r; j++) {
            mn = std::min(mn, data(i, j));
            mx = std::max(mx, data(i, j));
        }
        if (mx - mn > best_spread) {
            best_spread = mx - mn;
            best_dim = i;
        }
    }
    int dim = best_dim;
//    int dim = depth % d;
    std::sort(indices.begin() + l, indices.begin() + r + 1, [&](int a, int b) {
        return data(dim, a) < data(dim, b);
    });
    int m = (l + r) / 2;
    pNode node = make_node(dim, data(dim, indices[m]));
    assert(node < nodes.size());

    pNode subtree = build_tree(l, m, depth + 1);
    nodes[node].left = subtree;
    subtree = build_tree(m + 1, r, depth + 1);
    nodes[node].right = subtree;

    return node;
}

inline ftype sqr(ftype a) {
    return a * a;
}

int KDTree::find_nn(const dvector &x, ftype *best_dist_sqr, ftype margin_sqr,
                    const vec<int> &ignore) const {
    *best_dist_sqr = std::numeric_limits<ftype>::infinity();
    int best = -1;
    vec<ftype> partial_dist(d, 0);
    find_nn(x, head, &best, best_dist_sqr, margin_sqr, 0, ignore, partial_dist);
    return best;
}

void KDTree::find_nn(const dvector &x, KDTree::pNode node, int *best, ftype *best_dist_sqr,
                     ftype margin_sqr, ftype cur_bin_dist_sqr,
                     const vec<int> &ignore, vec<ftype> &partial_dist_sqr) const {
    if (margin_sqr >= 0 && *best_dist_sqr < margin_sqr) {
        return;
    }
    if (nodes[node].dim == -1) {
        // leaf node
        for (int i = nodes[node].lidx; i <= nodes[node].ridx; i++) {
            int index = indices[i];
            if (std::find(ignore.begin(), ignore.end(), index) == ignore.end()) {
                ftype dst_sqr = (x - data.col(index)).squaredNorm();
                assert(dst_sqr + 1e-5 >= cur_bin_dist_sqr);
                if (dst_sqr < *best_dist_sqr && (margin_sqr < 0 || dst_sqr <= margin_sqr)) {
                    *best_dist_sqr = dst_sqr;
                    *best = index;
                    if (margin_sqr >= 0) {
                        return;
                    }
                }
            }
        }
        return;
    }
    pNode fst, snd;
    if (x[nodes[node].dim] <= nodes[node].m) {
        fst = nodes[node].left;
        snd = nodes[node].right;
    } else {
        fst = nodes[node].right;
        snd = nodes[node].left;
    }
    if (fst >= 0) {
        find_nn(x, fst, best, best_dist_sqr, margin_sqr, cur_bin_dist_sqr, ignore,partial_dist_sqr);
    }
    ftype old_partial = partial_dist_sqr[nodes[node].dim];
    partial_dist_sqr[nodes[node].dim] = sqr(x[nodes[node].dim] - nodes[node].m);
    assert(old_partial <= partial_dist_sqr[nodes[node].dim]);
    ftype new_bin_dist_sqr = cur_bin_dist_sqr - old_partial + partial_dist_sqr[nodes[node].dim];
    if (snd >= 0 && new_bin_dist_sqr <= (margin_sqr < 0 ? *best_dist_sqr : margin_sqr)) {
        find_nn(x, snd, best, best_dist_sqr, margin_sqr, new_bin_dist_sqr, ignore,partial_dist_sqr);
    }
    partial_dist_sqr[nodes[node].dim] = old_partial;
}

KDTree::Node::Node(int lidx, int ridx) : lidx(lidx), ridx(ridx), dim(-1) {}
KDTree::Node::Node(int dim, ftype m) : dim(dim), m(m) {}
