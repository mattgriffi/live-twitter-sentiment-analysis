"""This class implements a fixed size queue. It will automatically delete elements at the front
of the queue to maintain the fixed size. It is also possible to create an iterator from it."""


class Queue:

    def __init__(self, size):
        self.max = size
        self.length = self.front = self.rear = 0
        self.q = [None] * self.max

    def size(self):
        """Returns the number of elements currently in the queue."""

        return self.length

    def empty(self):
        """Returns True if the queue is empty, else False."""

        return self.length == 0

    def enqueue(self, e):
        """Adds the element e to the end of the queue. If the queue has already reached its
        max size, the first element of the queue will be deleted to make room for e."""

        self.rear = (self.rear + 1) % self.max
        self.q[self.rear] = e
        if self.front == self.rear:
            self.front += 1
        self.length += 1

    def dequeue(self):
        """Returns the element at the front of the queue. Removes it from the queue."""
