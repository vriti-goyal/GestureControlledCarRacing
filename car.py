class Car:
    def __init__(self):
        self.lane = 1
        self.lane_x = [80, 170, 260]
        self.current_x = self.lane_x[self.lane]
        self.y = 420
        self.speed = 10

    def move_left(self):
        if self.lane > 0:
            self.lane -= 1

    def move_right(self):
        if self.lane < 2:
            self.lane += 1

    def get_position(self):
        target_x = self.lane_x[self.lane]

        if self.current_x < target_x:
            self.current_x = min(self.current_x + self.speed, target_x)
        elif self.current_x > target_x:
            self.current_x = max(self.current_x - self.speed, target_x)

        return self.current_x, self.y
