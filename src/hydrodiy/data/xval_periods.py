import numpy as np
import pandas as pd

class XvalPeriod():
    def __init__(self, start_active, end_active, warmup_years=5):
        self.warmup_years = int(warmup_years)
        self.start_active = pd.to_datetime(start_active)
        self.end_active = pd.to_datetime(end_active)
        self.start_total = self.start_active - pd.DateOffset(months=self.warmup_years * 12)
        self.end_total = self.end_active

    def __str__(self):
        return f"{self.start_active.date()} - {self.end_active.date()}"

    def index_active(self, time):
        return (time >= self.start_active) & (time <= self.end_active)

    def index_total(self, time):
        return (time >= self.start_total) & (time <= self.end_total)

    @property
    def nyears_active(self):
        delta = self.end_active - self.start_active
        return delta.days / 365.25

    @property
    def nyears_total(self):
        delta = self.end_total - self.start_total
        return delta.days / 365.25


class XvalPeriodSet():
    def __init__(self,
                 year_start,
                 year_end,
                 number_of_periods=2,
                 warmup_years=5,
                 wateryear_start=7,
                 timestep="day"):

        self.warmup_years = int(warmup_years)
        self.wateryear_start = int(wateryear_start)
        if timestep not in ["day", "month", "year"]:
            errmsg = f"Expected timestep in day/month/year got {timestep}."
            raise ValueError(errmsg)
        kw = {f"{timestep}s": 1}
        self.offset = pd.DateOffset(**kw)

        self.year_start = int(year_start)
        self.year_end = int(year_end)
        if self.year_start > self.year_end - 2:
            errmsg = f"Expected year_start({year_start})"\
                     + f"< year_end({year_end}) - 2."
            raise ValueError(errmsg)

        self._periods = {}
        self.number_of_periods = int(number_of_periods)
        self.generate_periods()

    def __len__(self):
        return len(self.periods)

    def __str__(self):
        txt = "Periods:"
        for pname, per in self.periods.items():
            txt += f"\n\t{pname}: {per} ({per.nyears_active:0.1f} years)"
        return txt

    @property
    def names(self):
        return list(self.periods.keys())

    @property
    def periods(self):
        return self._periods

    def generate_periods(self):
        self._periods = {}
        years = np.arange(self.year_start, self.year_end + 1)
        nyears = len(years)
        nper = self.number_of_periods
        nyears_per_period = nyears // nper
        ys = self.wateryear_start
        start = pd.to_datetime(f"{years[0]}-{ys:02d}-01")

        for iper in range(nper):
            end = start + pd.DateOffset(years=nyears_per_period) \
                - self.offset

            per = XvalPeriod(start, end,
                             warmup_years=self.warmup_years)
            pname = f"PER{iper + 1:02d}"
            self.periods[pname] = per

            # loop
            start = end + self.offset

    def get_validation_periods(self, pername):
        if pername not in self.periods:
            errmsg = f"Period {pername} is not in the list of periods."
            raise ValueError(errmsg)

        return {pname: per for pname, per in self.periods
                if pname != pername}

