#! /usr/bin/env python
import logging
import datetime
import fleming
import pytz
import os
from itertools import islice
import sys


class CSVLogger(object):  # pragma: no cover
    def __init__(self, name, log_file=None, level='info'):
        # create logger on the current module and set its level
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.INFO)
        self.logger.setLevel(getattr(logging, level.upper()))
        self.needs_header = True

        # create a formatter that creates a single line of json with a comma at the end
        self.formatter = logging.Formatter(
            (
                '%(created)s,%(name)s,"%(utc_time)s","%(eastern_time)s",%(levelname)s,"%(message)s"'
            )
        )

        self.log_file = log_file
        if self.log_file:
            # create a channel for handling the logger (stderr) and set its format
            ch = logging.FileHandler(log_file)
        else:
            # create a channel for handling the logger (stderr) and set its format
            ch = logging.StreamHandler()
        ch.setFormatter(self.formatter)

        # connect the logger to the channel
        self.logger.addHandler(ch)

    def log(self, msg, level='info'):
        HEADER = 'unix_time,module,utc_time,eastern_time,level,msg\n'
        if self.needs_header:
            if self.log_file and os.path.isfile(self.log_file):
                with open(self.log_file) as file_obj:
                    if len(list(islice(file_obj, 2))) > 0:
                        self.needs_header = False
                if self.needs_header:
                    with open(self.log_file, 'a') as file_obj:
                        file_obj.write(HEADER)
            else:
                if self.needs_header:
                    sys.stderr.write(HEADER)
            self.needs_header = False

        utc = datetime.datetime.utcnow()
        eastern = fleming.convert_to_tz(utc, pytz.timezone('US/Eastern'), return_naive=True)
        extra = {
            'utc_time': datetime.datetime.utcnow(),
            'eastern_time': eastern
        }
        func = getattr(self.logger, level)
        func(msg, extra=extra)