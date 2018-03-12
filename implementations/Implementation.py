class Implementation:
    """Default Implementation."""

    def __init__(self):
        """Intialize class."""
        pass

    def get_surrounding(self, row, col):
        if row > 0:
            above = [row -1, col]
        else:
            above = None

        if row < (len(self.imgdata) - 1):
            below = [row + 1, col]
        else:
            below = None

        if col > 0:
            left = [row, col - 1]
        else:
            left = None

        if col < (len(self.imgdata[row]) - 1):
            right = [row, col + 1]
        else:
            right = None

        return above,below,left,right
