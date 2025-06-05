import re
import subprocess
import tempfile

import pandas as pd
from rebench.interop.adapter import (
    GaugeAdapter,
    OutputNotParseable,
    ResultsIndicatedAsInvalid,
)
from rebench.model.data_point import DataPoint
from rebench.model.measurement import Measurement


class AlloyAdapter(GaugeAdapter):
    # To be sure about how to parse the output, give custom format
    # This avoids issues with perhaps none-standard /bin/usr/time
    time_format = '"max rss (kb): %M\nwall-time (secounds): %e\n"'
    re_formatted_time = re.compile(r"^wall-time \(secounds\): (\d+\.\d+)")
    re_formatted_rss = re.compile(r"^max rss \(kb\): (\d+)")

    _completed_time_availability_check = False
    _use_formatted_time = False
    _time_bin = None

    def __init__(self, include_faulty, executor):
        self.alloy_stats = tempfile.NamedTemporaryFile(delete=False)
        self.gc_stats = tempfile.NamedTemporaryFile(delete=False)

        GaugeAdapter.__init__(self, include_faulty, executor)

    def _create_command(self, command):
        if self._use_formatted_time:
            return "%s -f %s %s" % (self._time_bin, AlloyAdapter.time_format, command)
        else:
            # use standard, but without info on memory
            # TODO: add support for reading out memory info on OS X
            return "/usr/bin/time -p %s" % command

    def should_enable_premature_finalizer_optimization(self, run_id):
        exec = run_id.benchmark.suite.executor.name
        return "gcvs" in exec or "elision" in exec or "premopt-opt" in exec

    def should_enable_premature_finalizer_prevention(self, run_id):
        exec = run_id.benchmark.suite.executor.name
        return (
            "gcvs" in exec
            or "elision" in exec
            or "premopt-opt" in exec
            or "premopt-naive" in exec
        )

    def should_enable_finalizer_elision(self, run_id):
        exec = run_id.benchmark.suite.executor.name
        return "gcvs" in exec or "premopt" in exec or "elision-opt" in exec

    def setup_stats(self, run_id):
        run_id.env["ALLOY_LOG"] = self.alloy_stats.name
        run_id.env["DISPLAY"] = ":99"

    def acquire_command(self, run_id):
        self.setup_stats(run_id)
        command = run_id.cmdline_for_next_invocation()

        if not self._completed_time_availability_check:
            self._check_which_time_command_is_available()

        return self._create_command(command)

    def _check_which_time_command_is_available(self):
        time_bin = "/usr/bin/time"
        try:
            formatted_output = subprocess.call(
                ["/usr/bin/time", "-f", AlloyAdapter.time_format, "/bin/sleep", "0"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except OSError:
            formatted_output = 1

        if formatted_output == 1:
            try:
                formatted_output = subprocess.call(
                    [
                        "/opt/local/bin/gtime",
                        "-f",
                        AlloyAdapter.time_format,
                        "/bin/sleep",
                        "0",
                    ],
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                )
                if formatted_output == 0:
                    time_bin = "/opt/local/bin/gtime"
            except OSError:
                formatted_output = 1

        AlloyAdapter._use_formatted_time = formatted_output == 0
        AlloyAdapter._time_bin = time_bin
        AlloyAdapter._completed_time_availability_check = True

    def parse_stats(self, run_id, invocation, iteration, current):
        df = pd.read_csv(self.alloy_stats.name)
        if df.empty:
            print("The file is empty (no data rows).")
        else:
            stats = df.iloc[0]
            assert (
                self.should_enable_finalizer_elision(run_id) == stats["elision enabled"]
            )
            assert (
                self.should_enable_premature_finalizer_prevention(run_id)
                == stats["pfp enabled"]
            )
            assert (
                self.should_enable_premature_finalizer_optimization(run_id)
                == stats["premopt enabled"]
            )
            stats = stats[3:]
            for idx, value in stats.items():
                measure = Measurement(
                    invocation,
                    iteration,
                    float(value),
                    "stat",
                    run_id,
                    idx,
                )
                current.add_measurement(measure)

    def parse_data(self, data, run_id, invocation):
        iteration = 1
        data_points = []
        current = DataPoint(run_id)

        for line in data.split("\n"):
            if self.check_for_error(line):
                raise ResultsIndicatedAsInvalid(
                    "Output of bench program indicated error."
                )

            if self._use_formatted_time:
                match = self.re_formatted_time.match(line)
                if match:
                    self.parse_stats(run_id, invocation, iteration, current)
                    time = float(match.group(1)) * 1000
                    measure = Measurement(
                        invocation, iteration, time, "ms", run_id, "total"
                    )
                    current.add_measurement(measure)
                    data_points.append(current)
                    current = DataPoint(run_id)
                    iteration += 1
            else:
                raise NotImplementedError

        if not data_points:
            raise OutputNotParseable(data)

        return data_points


class AlloyMemAdapter(AlloyAdapter):
    def acquire_command(self, run_id):
        self.setup_stats(run_id)
        command = run_id.cmdline_for_next_invocation()
        return command
