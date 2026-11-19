import time


class Stopwatch:

  def __init__(self):
    self.start = time.perf_counter()

  def reset(self):
    self.start = time.perf_counter()

  @property
  def elapsed(self) -> str:
    elapsed_time = time.perf_counter() - self.start
    mins, secs = divmod(elapsed_time, 60)
    hours, mins = divmod(int(mins), 60)
    if hours:
        return f"{hours:02d}:{mins:02d}:{secs:05.2f}"
    return f"{int(mins):02d}:{secs:05.2f}"
