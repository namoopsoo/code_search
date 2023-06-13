import datetime
import pytz


def utc_ts():
    return datetime.datetime.utcnow(
        ).replace(tzinfo=pytz.UTC).strftime('%Y-%m-%dT%H%M%SZ')


