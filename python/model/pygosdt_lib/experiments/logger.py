from datetime import datetime

class Logger:
    def __init__(self, path=None, header=['time', 'message']):
        self.path = path if path != None else 'logs/{}.log'.format(str(datetime.now()))
        with open(self.path, 'w') as fp:
            fp.write(','.join(header) + '\n')

    def log(self, row):
        with open(self.path, 'a') as fp:
            fp.write(','.join([str(e) for e in row]) + '\n')
