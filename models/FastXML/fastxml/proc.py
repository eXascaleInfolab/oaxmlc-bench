from builtins import object
import multiprocessing
import traceback
import tempfile, pickle, traceback, os

class Result(object):
    def ready(self):
        raise NotImplementedError()
    def get(self):
        raise NotImplementedError()

class ForkResult(Result):
    def __init__(self, conn, p):
        self.conn = conn      # parent_conn
        self.p = p
        self._got = False
        self._val = None

    def ready(self):
        # True only when a message is available to recv()
        return self.conn.poll()

    def get(self, timeout=None):
        # If already received, return cached
        if self._got:
            return self._val

        # Wait for a message (optionally with timeout)
        if timeout is not None and not self.conn.poll(timeout):
            raise TimeoutError("child did not produce a result in time")

        if not self.conn.poll():
            # Child may have crashed without sending
            self.p.join(timeout=0)
            if not self.p.is_alive():
                raise RuntimeError("child exited without sending a result")
            # Otherwise still running; block until message arrives
            # (use with caution if you really want blocking)
        ok, payload = self.conn.recv()
        self.p.join()
        self.conn.close()

        if not ok:
            # Re-raise exception from child with its traceback
            err, tb = payload
            raise RuntimeError(f"child raised: {err}\n{tb}")

        self._got = True
        self._val = payload
        return payload

class SingleResult(Result):
    def __init__(self, res):
        self.res = res
    def ready(self):
        return True
    def get(self):
        return self.res

def _remote_call(conn, f, args):
    try:
        result = f(*args)  # potentially large object (Tree)
        # Persist to temp file; return only a small descriptor
        fd, path = tempfile.mkstemp(prefix="fastxml_", suffix=".pkl")
        os.close(fd)
        with open(path, "wb") as fh:
            pickle.dump(result, fh, protocol=pickle.HIGHEST_PROTOCOL)
        try:
            conn.send((True, {"path": path}))
        except (BrokenPipeError, OSError):
            # Parent died/closed early; nothing more to do
            pass
    except Exception as e:
        try:
            conn.send((False, (repr(e), traceback.format_exc())))
        except (BrokenPipeError, OSError):
            pass
    finally:
        try:
            conn.close()
        except Exception:
            pass

def faux_fork_call(f):
    def f2(*args):
        return SingleResult(f(*args))
    return f2

def fork_call(f):
    def f2(*args):
        parent_conn, child_conn = multiprocessing.Pipe(duplex=False)
        p = multiprocessing.Process(target=_remote_call, args=(child_conn, f, args))
        p.start()
        child_conn.close()  # close child end in parent
        return ForkResult(parent_conn, p)
    return f2