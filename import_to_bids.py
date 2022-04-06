# Authors: Teon Brooks <teon.brooks@gmail.com>
#          Stefan Appelhoff <stefan.appelhoff@mailbox.org>
#
# Modified: Mathias Sablé-Meyer
# License: BSD-3-Clause

import os
import warnings

from datetime import datetime

from mne import set_log_file
from mne_bids import write_raw_bids, BIDSPath
from mne_bids.read import _read_raw

import fire
import config_import


class BadConfig(Exception):
    def __init__(self, message="Something contradictory was required"):
        self.message = message
        super().__init__(self.message)


def main(
    project,
    root_name="bids_data",
    exclude=[],
    include=None,
    empty_rooms=True,
):
    project_path = f"./{project}"
    if isinstance(exclude, str):
        exclude = [exclude]
    if include is not None and set(include) & set(exclude):
        raise BadConfig("Some subjects are both included and excluded")
    for subject in os.scandir(project_path):
        if include is not None and subject.name not in include:
            print(f"- Ignoring {subject.name=}")
            continue
        if subject.name in exclude:
            print(f"- Ignoring {subject.name=}")
            continue
        if not os.path.exists("logs"):
            os.makedirs("logs")

        print(f"- Importing data from {subject.name=}")
        runs = None
        print(subject.path)
        sessions = os.scandir(subject.path)
        sessions_map = {
            s: (idx + 1) for idx, s in enumerate([x.name for x in sessions])
        }
        n_sessions = len(sessions_map.keys())
        sessions = os.scandir(subject.path)
        runs = [(session, run) for session in sessions for run in os.scandir(session)]
        for session, run in runs:
            if "emptyroom" in run.name:
                print(f"  - Ignoring run {run.name=} (EMPTYROOM)")
                continue

            date = datetime.now().isoformat()
            log_fname = None

            session_name = None
            if n_sessions > 1:
                session_name = f"{sessions_map[session.name]:02}"
                log_fname = f"logs/{date}_{subject.name}_{session_name}_{run.name}.log"
                print(
                    f"  - Importing data from {run.name=} of {session_name=} of {subject.name=}"
                )
            else:
                print(f"  - Importing data from {run.name=} of {subject.name=}")
                log_fname = f"logs/{date}_{subject.name}_{run.name}.log"
            print(f"    Check the log `{log_fname}` for warnings!")
            set_log_file(log_fname)

            raw = None
            # This is required to even touch non maxFiltered data
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw = _read_raw(run.path, allow_maxshield=True, verbose=False)
            raw.info["line_freq"] = 50

            run_number = config_import.parse_run(subject.name, run.name)
            task_name = config_import.parse_task(run.name)

            subject_safe_name = subject.name
            try:
                subject_safe_name = config_import.participants_mapping[
                    subject_safe_name
                ]
            except AttributeError:
                subject_safe_name = subject_safe_name.replace("_", "")
                subject_safe_name = subject_safe_name.replace("-", "")
                subject_safe_name = subject_safe_name.replace("/", "")

            bids_path = BIDSPath(
                subject=subject_safe_name,
                session=session_name,
                run=run_number,
                task=task_name,
                root=root_name,
            )

            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                raw.preload = False
                write_raw_bids(
                    raw,
                    bids_path,
                    event_id=config_import.event_id,
                    events_data=config_import.get_events(
                        raw, subject.name, session_name, run.name
                    ),
                    overwrite=True,
                    format="FIF",
                    verbose=False,
                )
        print(f"  Done with {subject.name=}")
        print("")
    if empty_rooms:
        for er_fname in config_import.empty_rooms:
            er_raw = _read_raw(f"./{project}/{er_fname}", allow_maxshield=True)
            er_date = er_raw.info["meas_date"].strftime("%Y%m%d")
            er_bids_path = BIDSPath(
                subject="emptyroom", session=er_date, task="noise", root=root_name
            )
            if not os.path.isfile(er_bids_path):
                er_raw.info["line_freq"] = 50
                write_raw_bids(er_raw, er_bids_path, overwrite=True)


if __name__ == "__main__":
    fire.Fire(main)
