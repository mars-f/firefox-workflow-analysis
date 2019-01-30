# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
Extract data about Mozilla's engineering workflow from various tool systems.
"""

import pandas as pd
import numpy as np

import requests

import tqdm
from mozautomation import commitparser

# Progress bars!
tqdm.tqdm.pandas()

# Pulling data from a local mirror of the mozilla-central repository is much faster!
# Run 'hg serve -p 9999' in your local Firefox clone of mozilla-central to set this up.
LOCAL_HG_SERVER = "http://localhost:9999"

# Some searches need to hit hg.mozilla.org directly because hg.m.o has custom plugins.
HGMO_URL = "https://hg.mozilla.org/"


def get_pushes_in_range(from_changeset, to_changeset):
    """Return a list of pushes between changesets X and Y."""
    # See https://mozilla-version-control-tools.readthedocs.io/en/latest/hgmo/pushlog.html#hgweb-commands for the URL structure.
    url = f"{HGMO_URL}/mozilla-central/json-pushes/?fromchange={from_changeset}&tochange={to_changeset}&version=2"
    r = requests.get(url)
    r.raise_for_status()

    # Munge the push structure from a nested structure with push IDs as dict keys to
    # a list of pushes where the push ID is a value.
    push_id_dict = r.json()["pushes"]
    flattened_pushes = []
    for pushid, pushdata in push_id_dict.items():
        newdata = dict(pushid=pushid, data=pushdata)
        flattened_pushes.append(newdata)
    return flattened_pushes


def fetch_rev_summary(rev_id):
    """Return the commit summary for the given changeset ID."""
    r = requests.get(f"{LOCAL_HG_SERVER}/json-rev/{rev_id}")
    r.raise_for_status()
    return r.json()["desc"]


def parse_bug_id(desc):
    """Parse a Firefox VCS commit message and return the bug ID, if possible."""
    ids = commitparser.parse_bugs(desc)
    if ids:
        # Leave it as a string.  Multiple bug id#s doesn't happen in practice.
        return str(ids[0])
    else:
        return np.NaN


def fetch_bug_creation_time(bug_id):
    """Fetch a bug's creation time from BMO."""
    r = requests.get(f"https://bugzilla.mozilla.org/rest/bug/{bug_id}")
    if not r.ok:
        return np.NaN
    return pd.to_datetime(r.json()["bugs"][0]["creation_time"])


def main():
    nightly_build_id = "20190122094123"
    nightly_publish_time = "2019-01-22T12:16:52Z"
    nightly_changeset_id = "f0c23db0d035dbe81e23eb4d619e493e38582d24"
    prev_nightly_changeset_id = "44369796f148630ff496be99f77a5eeea41c7d23"

    pushes = get_pushes_in_range(prev_nightly_changeset_id, nightly_changeset_id)

    # Get a list of all changesets in this nightly build
    df = pd.io.json.json_normalize(
        pushes,
        record_path=["data", "changesets"],
        meta=["pushid", ["data", "date"]],
        meta_prefix="changeset_",
        sep="_",
    )
    # Fix the name of the first column that json_normalize() gave us.
    df = df.rename(columns={0: "changeset"})

    # Convert push dates from Unix time to datetime
    df["changeset_pushtime"] = df["changeset_data_date"].apply(
        lambda t: pd.to_datetime(t, unit="s")
    )
    df = df.drop(columns="changeset_data_date")

    # Add our nightly build metadata
    df["nightly_build_id"] = nightly_build_id
    df["nightly_publish_time"] = pd.to_datetime(nightly_publish_time)

    # Add changeset summaries
    print("Getting changeset summaries")
    df["changeset_desc"] = df["changeset"].progress_apply(fetch_rev_summary)

    # Re-order the columns so 'changeset_' values are together
    df = df.sort_index(axis=1)

    # Split out the bug IDs
    df["bug"] = df["changeset_desc"].apply(parse_bug_id)

    # Drop all commits with no bug number
    commits_with_bugs = df[df["bug"].notna()]
    dropped = len(df) - len(commits_with_bugs)
    print(f"Dropped {dropped} commits with no associated bug number")

    df = commits_with_bugs

    # Add bug creation times
    print("Fetching bug creation times")
    df["bug_creation_time"] = df["bug"].progress_apply(fetch_bug_creation_time)

    # Not all bugs are publicly visible.  We won't have data for these bugs.
    public_bugs = df[df["bug_creation_time"].notna()]
    dropped = len(df) - len(public_bugs)
    print(f"Dropped {dropped} bugs that we couldn't access the data for")

    df = public_bugs

    print()
    print(f"Processed {df.shape[0]} records with {df.shape[1]} columns")
    print(f"Columns: {df.columns.to_list()}")

    outfile_name = "output.parq"
    df.to_parquet(outfile_name)
    print(f"Wrote {outfile_name}")


if __name__ == "__main__":
    main()
