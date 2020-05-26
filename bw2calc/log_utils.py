from datetime import datetime
import json
import logging
import os
import uuid

"""Adapted from json-log-formatter (https://github.com/marselester/json-log-formatter)

The MIT License (MIT)

Copyright (c) 2015 Marsel Mavletkulov

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE."""

BUILTIN_ATTRS = {
    "args",
    "asctime",
    "created",
    "exc_info",
    "exc_text",
    "filename",
    "funcName",
    "levelname",
    "levelno",
    "lineno",
    "module",
    "msecs",
    "message",
    "msg",
    "name",
    "pathname",
    "process",
    "processName",
    "relativeCreated",
    "stack_info",
    "thread",
    "threadName",
}


class JSONFormatter(logging.Formatter):
    json_lib = json

    def format(self, record):
        message = record.getMessage()
        extra = self.extra_from_record(record)
        json_record = self.json_record(message, extra, record)
        return self.to_json(json_record)

    def to_json(self, record):
        return self.json_lib.dumps(record, ensure_ascii=False)

    def extra_from_record(self, record):
        return {
            attr_name: record.__dict__[attr_name]
            for attr_name in record.__dict__
            if attr_name not in BUILTIN_ATTRS
        }

    def json_record(self, message, extra, record):
        if extra:
            data = extra
            if message:
                data["message"] = message
        else:
            # Also support logger.info({'foo': 'bar'})
            data = eval(message)
        data["time"] = str(datetime.utcnow())

        if record.exc_info:
            data["exc_info"] = self.formatException(record.exc_info)
        return data


def create_logger(dirpath=None, name=None, **kwargs):
    """Create a ``logger`` instance named ``bw2calc`` that can be used to log calculations.

    ``dirpath`` is the directory where the log file is saved. If ``dirpath`` is ``None``, no logger is created.

    ``name`` is the name of the calculation run, used to construct the log filepath.

    You can add other types of loggers, just add another handler to the ``bw2calc`` named logger before starting your calculations.

    Returns the filepath of the created log file.

    TODO: Decide on whether we copy safe_filepath to this package or create a common core package."""
    if dirpath is None:
        return

    assert os.path.isdir(dirpath) and os.access(dirpath, os.W_OK)

    # Use safe_filepath here
    filename = "{}.{}.json".format(name or uuid.uuid4().hex, str(datetime.utcnow()))

    formatter = JSONFormatter()
    fp = os.path.abspath(os.path.join(dirpath, filename))

    json_handler = logging.FileHandler(filename=fp, encoding="utf-8")
    json_handler.setFormatter(formatter)

    logger = logging.getLogger("bw2calc")
    logger.addHandler(json_handler)
    logger.setLevel(logging.INFO)

    return fp
