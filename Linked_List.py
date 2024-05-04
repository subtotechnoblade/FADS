class Node(object):
    __slots__ = "snapshot", "next", "parent"

    def __init__(self, snapshot, parent):
        self.snapshot = snapshot
        self.next = None
        self.parent = parent


class Linked_List:
    def __init__(self):
        self.start = None
        self.end = None

        # points to the current
        self.pointer = None

    def Add(self, snapshot):
        if self.start is None:
            self.start = Node(snapshot=snapshot, parent=None)
            self.end = self.start
            self.pointer = self.end
            return

        child = Node(snapshot, parent=self.pointer)
        self.pointer.next = child
        self.end = child
        self.pointer = child

    def Move_Pointer(self, inc):
        if inc < 0:
            if self.pointer.parent is not None:
                self.pointer = self.pointer.parent
        else:
            if self.pointer.next is not None:
                self.pointer = self.pointer.next
