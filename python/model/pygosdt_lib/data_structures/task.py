class Task:
    def __init__(self, priority, capture, path):
        self.priority = priority
        self.capture = capture
        self.path = path

    def key(self):
        return (self.capture, self.path)

    def __eq__(self, task):
        if type(task) != Task:
            return False
        return self.capture == task.capture and self.path == task.path

    def __le__(self, task):
        self.priority <= task.priority
    
    def __lt__(self, task):
        self.priority < task.priority

    def __ge__(self, task):
        self.priority >= task.priority

    def __gt__(self, task):
        self.priority > task.priority

    def __str__(self):
        return 'Task(priority={}, capture={}, path={})'.format(self.priority, self.capture, self.path)