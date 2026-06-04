#
# @lc app=leetcode id=3161 lang=python
#
# [3161] Block Placement Queries
#

# @lc code=start


class SegTree:
    def __init__(self, n):
        self.n = n
        # store max_gap, pref, suf for each node
        self.max_gap = [0] * (4 * n)
        self.pref = [0] * (4 * n)
        self.suf = [0] * (4 * n)
        self._build(1, 0, n)

    def _build(self, node, l, r):
        self.pref[node] = r + 1
        self.suf[node] = l - 1
        self.max_gap[node] = 0
        if l == r:
            return
        mid = (l + r) // 2
        self._build(node * 2, l, mid)
        self._build(node * 2 + 1, mid + 1, r)

    def _push_up(self, node, l, r):
        lc, rc = node * 2, node * 2 + 1
        self.pref[node] = self.pref[lc]
        self.suf[node] = self.suf[rc]
        cross = self.pref[rc] - self.suf[lc]
        self.max_gap[node] = max(self.max_gap[lc], self.max_gap[rc], cross)

    def update(self, node, l, r, x):
        if l == r:
            self.pref[node] = x
            self.suf[node] = x
            return
        mid = (l + r) // 2
        if x <= mid:
            self.update(node * 2, l, mid, x)
        else:
            self.update(node * 2 + 1, mid + 1, r, x)
        self._push_up(node, l, r)

    def query(self, node, l, r, ql, qr):
        if ql <= l and r <= qr:
            return self.max_gap[node], self.pref[node], self.suf[node]
        mid = (l + r) // 2
        if qr <= mid:
            return self.query(node * 2, l, mid, ql, qr)
        if ql > mid:
            return self.query(node * 2 + 1, mid + 1, r, ql, qr)
        lg, lp, ls = self.query(node * 2, l, mid, ql, qr)
        rg, rp, rs = self.query(node * 2 + 1, mid + 1, r, ql, qr)
        cross = rp - ls
        return max(lg, rg, cross), lp, rs


class Solution(object):
    def getResults(self, queries):
        """
        :type queries: List[List[int]]
        :rtype: List[bool]
        """
        MAX_X = 300001
        st = SegTree(MAX_X)
        st.update(1, 0, MAX_X, 0)  # virtual obstacle at 0
        results = []
        for q in queries:
            if q[0] == 1:
                st.update(1, 0, MAX_X, q[1])
            else:
                x, sz = q[1], q[2]
                max_gap, _, suf = st.query(1, 0, MAX_X, 0, x)
                tail_gap = x - suf  # gap from last obstacle to x
                results.append(max(max_gap, tail_gap) >= sz)
        return results


# @lc code=end


## Segment Tree Design
