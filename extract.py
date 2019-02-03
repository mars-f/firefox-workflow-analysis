# This Source Code Form is subject to the terms of the Mozilla Public
# License, v. 2.0. If a copy of the MPL was not distributed with this
# file, You can obtain one at http://mozilla.org/MPL/2.0/.
"""
Extract data about Mozilla's engineering workflow from various tool systems.
"""

import logging
import sys
from itertools import islice

import pandas as pd
import numpy as np
import requests
import tqdm
from mozautomation import commitparser
from kinto_http import Client as KintoClient

log = logging.getLogger()

# Progress bars!
tqdm.tqdm.pandas()

# Pulling data from a local mirror of the mozilla-central repository is much faster!
# Run 'hg serve -p 9999' in your local Firefox clone of mozilla-central to set this up.
LOCAL_HG_SERVER = "http://localhost:9999"

# Some searches need to hit hg.mozilla.org directly because hg.m.o has custom plugins.
HGMO_URL = "https://hg.mozilla.org/"

# Buildhub service, which holds all Firefox build artifact data.
BUILDHUB_URL = "https://buildhub.prod.mozaws.net/v1"

# We use the ActiveData warehouse service to give us bug records.  Very snappy.
ACTIVEDATA_URL = "https://activedata-public.devsvcprod.mozaws.net/query"

# Use a requests session to store default values
http = requests.Session()
# Set a friendly User Agent string
http.headers.update({"User-Agent": "firefox-eng-metrics mars@mozilla.com"})


def get_nightly_builds(from_datestr="2018-10", to_datestr="2018-11"):
    """Return nightly build data for all builds."""
    client = KintoClient(server_url=BUILDHUB_URL)
    records = client.get_records(
        **{
            "target.platform": "linux-x86_64",
            "target.channel": "nightly",
            "source.product": "firefox",
            "target.locale": "en-US",
            # Caution: use build.date because download.date is crazy for dates before
            # 2016.
            "gt_build.date": from_datestr,
            "lt_build.date": to_datestr,
            "_sort": "-download.date",
            "_fields": "build.id,source.revision,download.date",
            # "_limit": 10,
        },
        bucket="build-hub",
        collection="releases",
    )
    return records


def get_pushes_in_range(from_changeset, to_changeset):
    """Return a list of pushes between changesets X and Y."""
    # See https://mozilla-version-control-tools.readthedocs.io/en/latest/hgmo/pushlog.html#hgweb-commands for the URL structure.
    url = f"{HGMO_URL}/mozilla-central/json-pushes/?fromchange={from_changeset}&tochange={to_changeset}&version=2"
    r = http.get(url)
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
    r = http.get(f"{LOCAL_HG_SERVER}/json-rev/{rev_id}")
    r.raise_for_status()
    return r.json()["desc"]


def parse_bug_id(desc):
    """Parse a Firefox VCS commit message and return the bug ID, if possible."""
    ids = commitparser.parse_bugs(desc)
    if ids:
        # Multiple bug id#s doesn't happen in practice.
        return int(ids[0])
    else:
        return np.NaN


def fetch_bug_creation_times(bug_ids):
    """Return a DataFrame of bug ids and their creation times."""
    # See https://github.com/mozilla/ActiveData/blob/dev/docs/jx_tutorial.md for the
    # syntax.
    query = {
        "from": "public_bugs",
        "select": {"aggregate": "min", "value": "created_ts"},
        "groupby": "bug_id",
        "where": {"in": {"bug_id": list(bug_ids)}},
        "limit": len(bug_ids),
        "format": "table",
    }
    # Note that the bug list we ask for is allowed to be large.  The upper limit on
    # ActiveData responses is 50K records!
    r = http.post(ACTIVEDATA_URL, json=query)
    if not r.ok:
        raise RetrievalError(r.content)

    df = pd.DataFrame(r.json()["data"], columns=["bug", "creation_ts"])

    # Convert to Pandas datetimes
    df["bug_creation_time"] = pd.to_datetime(df["creation_ts"], unit="ms")
    df = df.drop(columns="creation_ts")

    return df


def get_data_for_commit_range(nightly_changeset_id, prev_nightly_changeset_id):
    """Return a DataFrame with workflow data between commits X and Y."""

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

    if df.empty:
        raise RetrievalError(
            f"No push data could be retrieved for "
            f"{nightly_changeset_id} to {prev_nightly_changeset_id}"
        )

    # Convert push dates from Unix time to datetime
    df["changeset_pushtime"] = df["changeset_data_date"].apply(
        lambda t: pd.to_datetime(t, unit="s")
    )
    df = df.drop(columns="changeset_data_date")

    # Add changeset summaries
    log.debug("Getting changeset summaries")
    tqdm.tqdm.pandas(desc="commit summaries")
    df["changeset_desc"] = df["changeset"].progress_apply(fetch_rev_summary)

    # Re-order the columns so 'changeset_' values are together
    df = df.sort_index(axis=1)

    # Split out the bug IDs
    df["bug"] = df["changeset_desc"].apply(parse_bug_id)

    # Drop all commits with no bug number
    commits_with_bugs = df[df["bug"].notna()]
    dropped = len(df) - len(commits_with_bugs)
    log.debug(f"Dropped {dropped} commits with no associated bug number")

    df = commits_with_bugs

    if df.empty:
        raise RetrievalError("No public bugs could be found for this build")

    # Add bug creation times
    log.debug("Fetching bug creation times")
    creation_times = fetch_bug_creation_times(df["bug"])
    df = pd.merge(df, creation_times, on="bug")

    # Not all bugs are publicly visible.  We won't have data for these bugs.
    public_bugs = df[df["bug_creation_time"].notna()]
    dropped = len(df) - len(public_bugs)
    log.debug(f"Dropped {dropped} bugs that we couldn't access the data for")

    df = public_bugs

    return df


def main():
    logging.basicConfig(stream=sys.stdout, level=logging.WARNING)

    df = pd.DataFrame()

    from_date = "2016-02"
    to_date = "2016-03"

    print(f"Process builds from {from_date} to {to_date}")
    builds = get_nightly_builds(from_date, to_date)
    build_count = len(builds)

    print(f"Found {build_count} builds")
    print("Extracting build data from sources")

    for target_build, prev_build in tqdm.tqdm(
        zip(islice(builds, 0, None), islice(builds, 1, None)),
        desc="builds",
        total=build_count,
    ):
        build_id = target_build["build"]["id"]
        build_rev = target_build["source"]["revision"]
        prev_rev = prev_build["source"]["revision"]

        log.debug("Processing build", build_id)
        try:
            cd = get_data_for_commit_range(build_rev, prev_rev)
        except RetrievalError as e:
            log.info("Skipping build %s" % target_build)
            log.info("Reason: %s" % e)
            continue

        cd["nightly_build_id"] = build_id
        cd["nightly_publish_time"] = pd.to_datetime(target_build["download"]["date"])
        df = pd.concat([df, cd])

    print()
    print(f"Processed {df.shape[0]} records with {df.shape[1]} columns")
    print(f"Columns: {df.columns.to_list()}")

    outfile_name = "output.parq"
    df.to_parquet(outfile_name)
    print(f"Wrote {outfile_name}")


class RetrievalError(Exception):
    pass


if __name__ == "__main__":
    main()
