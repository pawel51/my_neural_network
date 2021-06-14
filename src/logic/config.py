import threading

WEIGHTS_LOCK = threading.Lock()
WEIGHTS = None