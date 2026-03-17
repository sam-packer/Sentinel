# This file was developed with the assistance of Claude Code and Opus 4.6.

"""Rate limiter instance for the Sentinel API."""

from flask_limiter import Limiter
from flask_limiter.util import get_remote_address

limiter = Limiter(get_remote_address)
