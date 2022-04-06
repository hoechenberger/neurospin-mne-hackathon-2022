import re

import numpy as np
import pandas as pd

from mne import find_events


class BadConfig(Exception):
    def __init__(self, message="Something contradictory was required"):
        self.message = message
        super().__init__(self.message)


def get_events(raw):
    # Discretize MISC based on a threshold, and construct events from that,
    # first by counting the ramp ups, then by only considering those that are
    # more or less 1s apart (little danger of shift as we then compare exact
    # times with the channels)
    #
    # Note : the cutoff value for the photodiode was magically chosen by looking
    # a lot at the data. I'm sure there's a better way to find a cutoff point,
    # but do to refresh properties (i.e. sometimes the value "falls" during a
    # presentation) and overall sensitivity (from positioning of the
    # photodiode), it's not that straightforward
    # replace_idx = raw.info["ch_names"].index("MISC005")
    old_sti = raw.get_data(picks="MISC005")[0, :]
    new_sti = 5 * (old_sti > 0.10)

    my_misc = []
    last_was_up = False
    for idx, x in enumerate(new_sti):
        if x > 2.5 and not last_was_up:
            last_was_up = True
            my_misc.append([raw.first_samp + idx, 0, 5])
        elif x < 2.5:
            last_was_up = False

    # The delay is set to 4 frames, our worst case scenario: the first stimulus
    # was 2 frames "earlier" than expected, and the second was 2 frames "later".
    # Anything more than that is probably mad.
    fixed_misc = []
    for idx in range(len(my_misc) - 1):
        delta = my_misc[idx + 1][0] - my_misc[idx][0]
        if abs(delta - 1000) < 3.5 * 16:
            fixed_misc.append(my_misc[idx])
    fixed_misc.append(my_misc[len(my_misc) - 1])
    misc_events = np.array(fixed_misc)

    # STI101 is unrealiable: recompute it with powers of two
    events_list = []
    for idx, chan in enumerate([f"STI00{i}" for i in range(1, 8)]):
        events = find_events(
            raw,
            stim_channel=chan,
            consecutive=True,
            min_duration=0.032,
            shortest_event=0.032,
            initial_event=False,
        )
        df = pd.DataFrame(events)
        df.columns = ["time", "NA", "value"]
        df["value"] = (2 ** idx) * (df["value"] > 0)
        events_list.append(df)
    sti_101 = pd.concat(events_list).groupby(["time"]).sum().reset_index().values

    # Every once in a while, two triggers are shifted by a few ms and so two
    # events are created at different starting points. If that is the case,
    # merge them by summing them.
    new_sti_101 = []
    idx = 0
    while idx < len(sti_101) - 1:
        # If there's less than 10ms, merge
        if sti_101[idx + 1, 0] - sti_101[idx, 0] < 10:
            new_row = [sti_101[idx, 0], 0, sti_101[idx, 2] + sti_101[idx + 1, 2]]
            new_sti_101.append(new_row)
            idx = idx + 2
        else:
            new_sti_101.append(sti_101[idx, :])
            idx = idx + 1
    new_sti_101.append(sti_101[len(sti_101) - 1, :])
    sti_101 = np.array(new_sti_101)

    # Now merge while ensuring that events are close enough in time (150ms, to
    # survive delays introduced by frame misses and projector delay).
    merged_events = []
    for idx in range(int(len(sti_101))):
        if abs(misc_events[idx, 0] - sti_101[idx, 0]) <= 150:
            merged_events.append([misc_events[idx, 0], 0, sti_101[idx, 2]])
        else:
            assert False

    merged_events = np.array(merged_events)
    return merged_events


def parse_run(subject, raw_file_filename):
    m = re.search("run([0-9]+)", raw_file_filename)
    return int(m.group(0)[3:])


def parse_task(raw_file_filename):
    return "POGS"


participants_mapping = {
    "ll_180197": "09",
}

event_id = {
    "rectangle/reference": 8,
    "rectangle/outlier/1": 9,
    "rectangle/outlier/2": 10,
    "rectangle/outlier/3": 11,
    "rectangle/outlier/4": 12,
    "square/reference": 16,
    "square/outlier/1": 17,
    "square/outlier/2": 18,
    "square/outlier/3": 19,
    "square/outlier/4": 20,
    "isoTrapezoid/reference": 24,
    "isoTrapezoid/outlier/1": 25,
    "isoTrapezoid/outlier/2": 26,
    "isoTrapezoid/outlier/3": 27,
    "isoTrapezoid/outlier/4": 28,
    "parallelogram/reference": 32,
    "parallelogram/outlier/1": 33,
    "parallelogram/outlier/2": 34,
    "parallelogram/outlier/3": 35,
    "parallelogram/outlier/4": 36,
    "losange/reference": 40,
    "losange/outlier/1": 41,
    "losange/outlier/2": 42,
    "losange/outlier/3": 43,
    "losange/outlier/4": 44,
    "kite/reference": 48,
    "kite/outlier/1": 49,
    "kite/outlier/2": 50,
    "kite/outlier/3": 51,
    "kite/outlier/4": 52,
    "rightKite/reference": 56,
    "rightKite/outlier/1": 57,
    "rightKite/outlier/2": 58,
    "rightKite/outlier/3": 59,
    "rightKite/outlier/4": 60,
    "rustedHinge/reference": 64,
    "rustedHinge/outlier/1": 65,
    "rustedHinge/outlier/2": 66,
    "rustedHinge/outlier/3": 67,
    "rustedHinge/outlier/4": 68,
    "hinge/reference": 72,
    "hinge/outlier/1": 73,
    "hinge/outlier/2": 74,
    "hinge/outlier/3": 75,
    "hinge/outlier/4": 76,
    "trapezoid/reference": 80,
    "trapezoid/outlier/1": 81,
    "trapezoid/outlier/2": 82,
    "trapezoid/outlier/3": 83,
    "trapezoid/outlier/4": 84,
    "random/reference": 88,
    "random/outlier/1": 89,
    "random/outlier/2": 90,
    "random/outlier/3": 91,
    "random/outlier/4": 92,
}

empty_rooms = [
    "ll_180197/220217/emptyroom.fif",
]
