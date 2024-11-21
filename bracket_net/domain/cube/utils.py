from datetime import datetime
import os

class DebugLogger():
    def __init__(self):
        self.logdir = "cube_logs"
        # set datetime to log filename
        self.log_filename = f"{self.logdir}/cube_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        if not os.path.exists(self.logdir):
            os.makedirs(self.logdir)

    @staticmethod
    def get_instance():
        if not hasattr(DebugLogger, "_instance"):
            DebugLogger._instance = DebugLogger()
        return DebugLogger._instance

    def rint(self, *args, **kwargs):
        with open(self.log_filename, "a") as f:
            print(*args, **kwargs, file=f)