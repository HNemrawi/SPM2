"""Generate a deterministic synthetic HMIS-style CSV for the golden-output harness.

The CSV uses the raw HMIS export column headers (e.g. ``Clients Client ID``) so
that ``ColumnMapper`` and ``standardize_data_types`` are exercised end-to-end,
not bypassed. Numpy + a fixed seed make every run produce the same bytes.

Run:  python -m tests.golden.generate_sample
Output: tests/golden/sample_hmis.csv
"""

from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 20260513
N_CLIENTS = 1500
N_ROWS_TARGET = 6000
DATE_MIN = date(2021, 1, 1)
DATE_MAX = date(2025, 12, 31)
OUT = Path(__file__).parent / "sample_hmis.csv"

PROJECT_TYPES = [
    "Street Outreach",
    "Emergency Shelter – Entry Exit",
    "Emergency Shelter – Night-by-Night",
    "Transitional Housing",
    "Safe Haven",
    "PH – Housing Only",
    "PH – Housing with Services (no disability required for entry)",
    "PH – Permanent Supportive Housing (disability required for entry)",
    "PH – Rapid Re-Housing",
    "Homelessness Prevention",
    "Coordinated Entry",
    "Day Shelter",
    "Services Only",
]

PROJECT_TYPE_WEIGHTS = np.array(
    [0.10, 0.18, 0.08, 0.07, 0.03, 0.10, 0.06, 0.08, 0.12, 0.06, 0.05, 0.04, 0.03]
)
PROJECT_TYPE_WEIGHTS = PROJECT_TYPE_WEIGHTS / PROJECT_TYPE_WEIGHTS.sum()
PH_TYPES = {p for p in PROJECT_TYPES if p.startswith("PH")}

GENDERS = ["Woman", "Man", "Non-Binary", "Transgender", "Client doesn't know", "Data not collected"]
GENDER_WEIGHTS = [0.45, 0.46, 0.03, 0.02, 0.02, 0.02]

RACE_ETHN = [
    "White",
    "Black, African American, or African",
    "Hispanic/Latina/e/o",
    "American Indian, Alaska Native, or Indigenous",
    "Asian or Asian American",
    "Multi-Racial",
    "Native Hawaiian or Pacific Islander",
    "Client doesn't know",
    "Data not collected",
]
RACE_WEIGHTS = [0.36, 0.28, 0.18, 0.04, 0.03, 0.05, 0.02, 0.02, 0.02]

VETERAN = ["Yes", "No", "Client doesn't know", "Data not collected"]
VETERAN_WEIGHTS = [0.08, 0.86, 0.03, 0.03]

YES_NO = ["Yes", "No"]
YES_NO_DK = ["Yes", "No", "Client doesn't know", "Data not collected"]

PRIOR_LIVING = [
    "Homeless Situation",
    "Institutional Situation",
    "Temporary Housing Situation",
    "Permanent Housing Situation",
    "Other",
    "Data not collected",
]
PRIOR_LIVING_WEIGHTS = [0.55, 0.10, 0.18, 0.10, 0.04, 0.03]

HOUSEHOLD_TYPES = ["Single Adult", "Family", "Youth", "Couple"]
HOUSEHOLD_WEIGHTS = [0.62, 0.30, 0.05, 0.03]

AGE_TIERS = ["Under 18", "18 to 24", "25 to 34", "35 to 44", "45 to 54", "55 to 61", "62+"]
AGE_TIER_WEIGHTS = [0.05, 0.10, 0.20, 0.22, 0.22, 0.13, 0.08]

EXIT_DEST_CATS = [
    "Permanent Housing Situations",
    "Temporary Housing Situations",
    "Institutional Situations",
    "Homeless Situations",
    "Other",
    "No Exit Interview Completed",
]
EXIT_DEST_CAT_WEIGHTS = [0.32, 0.18, 0.10, 0.20, 0.10, 0.10]

EXIT_DESTINATIONS_BY_CAT = {
    "Permanent Housing Situations": [
        "Rental by client, no ongoing housing subsidy",
        "Rental by client, with ongoing housing subsidy",
        "Owned by client, no ongoing housing subsidy",
        "Staying or living with family, permanent tenure",
        "Permanent housing for formerly homeless",
    ],
    "Temporary Housing Situations": [
        "Transitional housing for homeless persons",
        "Staying or living with family, temporary tenure",
        "Staying or living with friends, temporary tenure",
        "Hotel or motel paid for without emergency shelter voucher",
    ],
    "Institutional Situations": [
        "Hospital or other residential non-psychiatric medical facility",
        "Jail, prison or juvenile detention facility",
        "Psychiatric hospital or other psychiatric facility",
        "Substance abuse treatment facility or detox center",
    ],
    "Homeless Situations": [
        "Place not meant for habitation",
        "Emergency shelter",
        "Safe Haven",
    ],
    "Other": ["Other", "Deceased", "Client refused"],
    "No Exit Interview Completed": ["No exit interview completed", "Data not collected"],
}

AGENCIES = [f"Agency {chr(ord('A') + i)}" for i in range(8)]
PROGRAMS_PER_AGENCY = 3
PROGRAMS = [f"{agency} — Program {p}" for agency in AGENCIES for p in range(1, PROGRAMS_PER_AGENCY + 1)]

COCS = ["CoC-001", "CoC-002", "CoC-003"]
LOCAL_COCS = ["Local-A", "Local-B", "Local-C", "Local-D"]


def _rand_date(rng: np.random.Generator, start: date, end: date) -> date:
    delta_days = (end - start).days
    return start + timedelta(days=int(rng.integers(0, delta_days + 1)))


def _draw_categorical(rng: np.random.Generator, choices: list, weights: list) -> str:
    return choices[int(rng.choice(len(choices), p=weights))]


def generate() -> pd.DataFrame:
    rng = np.random.default_rng(SEED)
    rows = []
    enroll_id = 1_000_000

    # Per-client immutable attributes
    client_pool = []
    for client_id in range(10_000, 10_000 + N_CLIENTS):
        dob = _rand_date(rng, date(1955, 1, 1), date(2010, 1, 1))
        client_pool.append(
            {
                "ClientID": client_id,
                "DOB": dob,
                "Gender": _draw_categorical(rng, GENDERS, GENDER_WEIGHTS),
                "Race": _draw_categorical(rng, RACE_ETHN, RACE_WEIGHTS),
                "Veteran": _draw_categorical(rng, VETERAN, VETERAN_WEIGHTS),
            }
        )

    avg_enrollments = N_ROWS_TARGET / N_CLIENTS
    for client in client_pool:
        n_enrolls = max(1, int(rng.poisson(avg_enrollments)))
        # Walk forward in time per client so multiple exits/returns interleave.
        cursor = _rand_date(rng, DATE_MIN, date(2024, 1, 1))
        for _ in range(n_enrolls):
            if cursor >= DATE_MAX:
                break
            start = cursor
            project_type = _draw_categorical(rng, PROJECT_TYPES, PROJECT_TYPE_WEIGHTS.tolist())
            # PH stays last longer; homeless services shorter.
            base_days = 240 if project_type in PH_TYPES else 60
            length = int(rng.exponential(base_days))
            length = min(length, (DATE_MAX - start).days)
            has_exit_prob = 0.85 if project_type in PH_TYPES else 0.92
            has_exit = rng.random() < has_exit_prob and length > 0
            exit_date = start + timedelta(days=length) if has_exit else None

            agency = _draw_categorical(rng, AGENCIES, [1 / len(AGENCIES)] * len(AGENCIES))
            program = f"{agency} — Program {int(rng.integers(1, PROGRAMS_PER_AGENCY + 1))}"
            coc = _draw_categorical(rng, COCS, [1 / len(COCS)] * len(COCS))
            local_coc = _draw_categorical(rng, LOCAL_COCS, [1 / len(LOCAL_COCS)] * len(LOCAL_COCS))

            if has_exit:
                cat = _draw_categorical(rng, EXIT_DEST_CATS, EXIT_DEST_CAT_WEIGHTS)
                dest = _draw_categorical(
                    rng,
                    EXIT_DESTINATIONS_BY_CAT[cat],
                    [1 / len(EXIT_DESTINATIONS_BY_CAT[cat])] * len(EXIT_DESTINATIONS_BY_CAT[cat]),
                )
            else:
                cat, dest = "", ""

            move_in = ""
            if project_type in PH_TYPES and has_exit and cat == "Permanent Housing Situations":
                offset = int(rng.integers(0, max(1, length // 2 + 1)))
                move_in = (start + timedelta(days=offset)).isoformat()

            row = {
                "Clients Unique Identifier": f"U{client['ClientID']:08d}",
                "Clients Client ID": client["ClientID"],
                "Clients Date of Birth Date": client["DOB"].isoformat(),
                "Clients Gender": client["Gender"],
                "Clients Race and Ethnicity": client["Race"],
                "Clients Veteran Status": client["Veteran"],
                "Entry Screen Age Tier": _draw_categorical(rng, AGE_TIERS, AGE_TIER_WEIGHTS),
                "Entry Screen Income from any Source": _draw_categorical(rng, YES_NO_DK, [0.30, 0.60, 0.05, 0.05]),
                "Entry Screen Any Disability": _draw_categorical(rng, YES_NO_DK, [0.40, 0.50, 0.05, 0.05]),
                "Entry Screen Head of Household (Yes / No)": _draw_categorical(rng, YES_NO, [0.70, 0.30]),
                "Entry Screen Currently Fleeing Domestic Violence": _draw_categorical(rng, YES_NO_DK, [0.08, 0.85, 0.04, 0.03]),
                "Entry Screen Prior Living Situation Category": _draw_categorical(rng, PRIOR_LIVING, PRIOR_LIVING_WEIGHTS),
                "Entry Screen Chronically Homeless Project Start - Household": _draw_categorical(rng, YES_NO_DK, [0.15, 0.75, 0.05, 0.05]),
                "Enrollments Enrollment ID": enroll_id,
                "Enrollments Household Type": _draw_categorical(rng, HOUSEHOLD_TYPES, HOUSEHOLD_WEIGHTS),
                "Enrollments Household Move-In Date": move_in,
                "Enrollments Project Start Date": start.isoformat(),
                "Enrollments Project Exit Date": exit_date.isoformat() if exit_date else "",
                "Enrollments Reporting Period Start Date": "2021-01-01",
                "Enrollments Reporting Period End Date": "2025-12-31",
                "Programs Agency Name": agency,
                "Programs Name": program,
                "Program Custom Local CoC Code": local_coc,
                "Programs Program Setup CoC": coc,
                "Programs Continuum Project": _draw_categorical(rng, YES_NO, [0.92, 0.08]),
                "Programs Project Type Code": project_type,
                "SSVF RRH": _draw_categorical(rng, ["SSVF RRH", "Not SSVF RRH"], [0.05, 0.95]),
                "Update/Exit Screen Destination Category": cat,
                "Update/Exit Screen Destination": dest,
            }
            rows.append(row)
            enroll_id += 1
            # Gap between this enrollment ending and the next starting.
            next_start_offset = length + int(rng.integers(7, 200))
            cursor = start + timedelta(days=next_start_offset)

    df = pd.DataFrame(rows)
    return df


def main() -> None:
    df = generate()
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")
    print(f"Unique clients: {df['Clients Client ID'].nunique()}")
    print(f"Project type distribution:")
    print(df["Programs Project Type Code"].value_counts().to_string())


if __name__ == "__main__":
    main()
