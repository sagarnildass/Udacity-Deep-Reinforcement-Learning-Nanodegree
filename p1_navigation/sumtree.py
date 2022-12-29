import numpy


class SumTree:
    '''
    A binary sum-tree. See Appendix B.2.1. in https://arxiv.org/pdf/1511.05952.pdf

    Adapted from implementation at:
    https://github.com/jaromiru/AI-blog/blob/master/SumTree.py
    '''

    def __init__(self, capacity):
        self.write = 0
        self.capacity = capacity
        self.tree = numpy.zeros(2 * capacity - 1)
        self.data = numpy.zeros(capacity, dtype=object)
        self.n_entries = 0

    # update to the root node
    def _propagate(self, idx, change):
        """Propagate priority update up through the tree

        Arguments:
            idx {int} -- index to change
            change {float} -- priority change to propagate
        """

        parent = (idx - 1) // 2
        self.tree[parent] += change
        if parent != 0:
            self._propagate(parent, change)

    def _retrieve(self, idx, s):
        """Retrieve sample on lead node

        Arguments:
            idx {int} -- index in tree
            s {float} -- value to sample

        Returns:
            [int] -- index of sample
        """

        left = 2 * idx + 1
        right = left + 1
        if left >= len(self.tree):
            return idx
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s - self.tree[left])

    def total(self):
        """Value of root node

        Returns:
            [float] -- root node value
        """

        return self.tree[0]

    def add(self, p, data):
        """Add a priority & sample to the tree

        Arguments:
            p {float} -- Priority, i.e. TD error
            data {tuple} -- tuple of (state, action, reward, next_state, done)
        """

        idx = self.write + self.capacity - 1
        self.data[self.write] = data
        self.update(idx, p)
        self.write += 1
        if self.write >= self.capacity:
            self.write = 0
        if self.n_entries < self.capacity:
            self.n_entries += 1

    def update(self, idx, p):
        """Update the priority at a given index

        Arguments:
            idx {int} -- index of sample
            p {float} -- updated priority
        """

        change = p - self.tree[idx]
        self.tree[idx] = p
        self._propagate(idx, change)

    def get(self, s):
        """Get idx, priority & sample for value s

        Arguments:
            s {float} -- value to sample with

        Returns:
            [tuple] -- (index, priority, sample)
        """

        idx = self._retrieve(0, s)
        dataIdx = idx - self.capacity + 1
        return (idx, self.tree[idx], self.data[dataIdx])