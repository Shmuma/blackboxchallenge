import logging as log
import os

from time import time
from datetime import timedelta


class OptionLoader:
    """
    Checks for file modification and reloads options from it.
    """
    def __init__(self, file_name):
        self.file_name = file_name
        self.mtime = self.get_mtime()
        self.values = self.read_content()

    def get_mtime(self):
        """
        Return modification time for file, None if doesn't exists
        """
        if not os.path.exists(self.file_name):
            return None
        st = os.stat(self.file_name)
        return st.st_mtime

    def check(self):
        """
        Performs check of file modification and load it's values.
        :return: true if file was modified
        """
        mtime = self.get_mtime()
        if mtime == self.mtime:
            return False
        if mtime is None:
            log.info("Option file disappeared")
        elif self.mtime is None:
            log.info("Option file has been created")
        else:
            log.info("Option file was modified {age} ago, passed time={passed}".format(
                age=timedelta(seconds=time() - mtime),
                passed=timedelta(seconds=mtime - self.mtime)
            ))
        self.mtime = mtime
        self.values = self.read_content()

    def read_content(self):
        res = {}

        if not os.path.exists(self.file_name):
            return res

        with open(self.file_name, "r") as fd:
            for l in fd:
                l = l.strip()
                if len(l) == 0:
                    continue
                v = map(lambda s: s.strip(), l.split("="))
                name, val = v[0], long(v[1])
                res[name] = val

        return res
