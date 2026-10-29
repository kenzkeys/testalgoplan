"""Sleep until just inside the tt_moc decision window (15:45 to 15:58 New York time).

GitHub Actions schedules can start late, so the workflow triggers about 20 minutes early and
this waits. Outside 15:00 to 15:46 ET it returns at once, and the incubation script then only
reconciles. Standard library only, so it runs before any dependency is installed.
"""

from datetime import datetime, time, timedelta
from time import sleep
from zoneinfo import ZoneInfo

NY = ZoneInfo("America/New_York")
TARGET = time(15, 46)
EARLIEST = time(15, 0)

now = datetime.now(NY)
if EARLIEST <= now.time() < TARGET and now.weekday() < 5:
    until = now.replace(hour=TARGET.hour, minute=TARGET.minute, second=0, microsecond=0)
    secs = (until - now).total_seconds()
    print(f"{now:%H:%M:%S} ET, waiting {secs / 60:.1f} min for the decision window")
    sleep(max(0.0, secs))
else:
    print(f"{now:%a %H:%M} ET is not before the window today; running reconcile only")
