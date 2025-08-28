"""
UI Templates
"""

from typing import Optional

import streamlit as st


# ==================== NEUTRAL COLOR SYSTEM ====================
class NeutralPalette:
    """Enhanced neutral colors that adapt to theme context"""

    # Primary brand colors
    PRIMARY = "#0066CC"
    PRIMARY_HOVER = "#0052A3"

    # Adaptive backgrounds using currentColor
    BG_SUBTLE = "color-mix(in srgb, currentColor 5%, transparent)"
    BG_LIGHT = "color-mix(in srgb, currentColor 8%, transparent)"
    BG_MEDIUM = "color-mix(in srgb, currentColor 12%, transparent)"

    # Borders that adapt to theme
    BORDER_SUBTLE = "color-mix(in srgb, currentColor 15%, transparent)"
    BORDER_DEFAULT = "color-mix(in srgb, currentColor 20%, transparent)"

    # Text opacity levels for hierarchy
    TEXT_PRIMARY = "1"
    TEXT_SECONDARY = "0.85"
    TEXT_MUTED = "0.65"

    # Semantic colors that work in both themes
    SUCCESS = "#059862"
    WARNING = "#D97706"
    DANGER = "#DC2626"
    INFO = "#0066CC"

    # Neutral colors for better adaptation
    NEUTRAL_HEADER = "currentColor"
    NEUTRAL_TEXT = "inherit"
    NEUTRAL_MUTED = "color-mix(in srgb, currentColor 70%, transparent)"
    NEUTRAL_LIGHT = "color-mix(in srgb, currentColor 50%, transparent)"
    NEUTRAL_BORDER = "color-mix(in srgb, currentColor 25%, transparent)"
    NEUTRAL_BG = "color-mix(in srgb, currentColor 10%, transparent)"
    NEUTRAL_BG_LIGHT = "color-mix(in srgb, currentColor 5%, transparent)"


# ==================== HEADER COMPONENTS ====================


def get_header_logo_html() -> str:
    """
    Enhanced header logo with better styling and hover effects.
    """
    return f"""
    <div style="
        background: {NeutralPalette.BG_SUBTLE};
        padding: 20px 28px;
        border-radius: 16px;
        border: 1px solid {
        NeutralPalette.BORDER_SUBTLE};
        text-align: left;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    "
    onmouseover="this.style.transform='translateY(-2px)'; this.style.boxShadow='0 4px 12px rgba(0, 0, 0, 0.1)';"
    onmouseout="this.style.transform='translateY(0)'; this.style.boxShadow='0 1px 3px rgba(0, 0, 0, 0.05)';">
        <a href="https://icalliances.org/" target="_blank" style="text-decoration: none;">
            <img src="https://images.squarespace-cdn.com/content/v1/54ca7491e4b000c4d5583d9c/eb7da336-e61c-4e0b-bbb5-1a7b9d45bff6/Dash+Logo+2.png?format=1000w"
                 width="350"
                 alt="DASH Logo"
                 style="max-width: 100%; height: auto; display: block;">
        </a>
    </div>
    """


# ==================== FOOTER COMPONENTS ====================


def get_footer_html() -> str:
    """
    Enhanced footer with improved spacing and visual design.
    """
    return f"""
    <div style="
        background: {NeutralPalette.BG_SUBTLE};
        padding: 48px 36px;
        border-radius: 16px;
        border: 1px solid {NeutralPalette.BORDER_SUBTLE};
        margin-top: 80px;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    ">
        <div style="margin-bottom: 36px;">
            <a href="https://icalliances.org/" target="_blank" style="
                display: inline-block;
                transition: transform 0.2s ease, opacity 0.2s ease;
            "
            onmouseover="this.style.transform='scale(1.05)'; this.style.opacity='1';"
            onmouseout="this.style.transform='scale(1)'; this.style.opacity='0.9';">
                <img src="https://images.squarespace-cdn.com/content/v1/54ca7491e4b000c4d5583d9c/eb7da336-e61c-4e0b-bbb5-1a7b9d45bff6/Dash+Logo+2.png?format=900w"
                     width="160"
                     alt="DASH Logo"
                     style="opacity: 0.9;">
            </a>
            <div style="
                margin-top: 16px;
                font-style: italic;
                opacity: {NeutralPalette.TEXT_MUTED};
                font-size: 0.95rem;
                letter-spacing: 0.01em;
            ">
                DASH™ is a trademark of Institute for Community Alliances.
            </div>
        </div>

        <div style="
            padding-top: 24px;
            border-top: 1px solid {NeutralPalette.BORDER_SUBTLE};
        ">
            <a href="https://icalliances.org/" target="_blank" style="
                display: inline-block;
                transition: transform 0.2s ease, opacity 0.2s ease;
            "
            onmouseover="this.style.transform='scale(1.05)'; this.style.opacity='0.9';"
            onmouseout="this.style.transform='scale(1)'; this.style.opacity='0.75';">
                <img src="https://images.squarespace-cdn.com/content/v1/54ca7491e4b000c4d5583d9c/1475614371395-KFTYP42QLJN0VD5V9VB1/ICA+Official+Logo+PNG+%28transparent%29.png?format=1500w"
                     width="120"
                     alt="ICA Logo"
                     style="opacity: 0.75;">
            </a>
            <div style="
                margin-top: 16px;
                opacity: {NeutralPalette.TEXT_MUTED};
                font-size: 0.875rem;
                letter-spacing: 0.025em;
            ">
                © 2025 Institute for Community Alliances (ICA). All rights reserved.
            </div>
        </div>
    </div>
    """


def render_footer():
    """Render the application footer"""
    st.html(get_footer_html())


# ==================== ORIGINAL ABOUT SECTIONS WITH NEUTRAL COLORS =======

ABOUT_SPM2_CONTENT = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: {NeutralPalette.NEUTRAL_TEXT}; line-height: 1.6;">
    <p style="margin-bottom: 20px;">
        <strong style="font-size: 1.1em; color: {NeutralPalette.NEUTRAL_HEADER};">SPM2 Analysis Overview</strong><br>
        <em style="color: {NeutralPalette.NEUTRAL_MUTED}
    ;">Assessing housing stability by tracking if and when clients return to homeless services after exiting to permanent housing.</em>
    </p>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">1. Default Client Universe</h3>
        <p style="margin-bottom: 10px;"><strong>Included Project Types:</strong></p>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li>Street Outreach (SO)</li>
            <li>Emergency Shelter (ES)</li>
            <li>Transitional Housing (TH)</li>
            <li>Safe Haven (SH)</li>
            <li>Permanent Housing (PH, PSH, RRH)</li>
        </ul>

        <p style="margin-bottom: 10px;"><strong>Date Window:</strong><br>
        Consider exits that occurred <strong>within 730 days before</strong> the reporting period</p>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li>Exit Date ≥ (Report Start Date – 730 days)</li>
            <li>Exit Date ≤ (Report End Date – 730 days)</li>
        </ul>

        <p style="background-color: rgba(217, 119, 6, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid {NeutralPalette.WARNING};">
            <strong>NOTE:</strong> Subject to change based on filter selections.
        </p>
    </div>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">2. Identifying the "Permanent Housing" Exit</h3>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li><strong>Definition:</strong> the client's <strong>earliest</strong> exit to any Permanent Housing destination</li>
            <li><strong>Tie‑Breaker:</strong> if multiple exits share that date, pick the one with <strong>lowest Enrollment ID</strong></li>
        </ul>

        <p style="background-color: rgba(217, 119, 6, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid {NeutralPalette.WARNING};">
            <strong>NOTE:</strong> Exit Filters are applied BEFORE we identify the earliest exit to permanent housing. So, if the earliest exit to permanent housing for this client is in a program type that has been filtered out, we will select the second-most recent exit to permanent housing, and so on.
        </p>
    </div>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">3. Scanning for a Return to Homeless Services</h3>
        <ol style="margin-left: 20px; margin-bottom: 15px;">
            <li style="margin-bottom: 8px;"><strong>Search Window:</strong> from the PH exit date up to the end of the reporting period</li>
            <li style="margin-bottom: 8px;"><strong>Eligible Return Enrollments:</strong>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li>Any SO, ES, TH, or SH project</li>
                    <li>PH project re‑entries where:
                        <ul style="margin-left: 20px; margin-top: 5px;">
                            <li>Start date <strong>&gt; 14 days</strong> after the original PH exit</li>
                            <li>Start date does not overlap with any other PH program enrollment</li>
                            <li>Start date is not within 14 days of exit from any other PH program exit</li>
                        </ul>
                    </li>
                </ul>
            </li>
            <li><strong>First Qualifying Return:</strong>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li>Project Start ≥ PH Exit Date</li>
                    <li>Project Start ≤ Report End Date</li>
                    <li>Stop at the very first enrollment that fulfills the above</li>
                </ul>
            </li>
        </ol>

        <p style="background-color: rgba(217, 119, 6, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid {NeutralPalette.WARNING};">
            <strong>NOTE:</strong> Return Filters are applied BEFORE we begin scanning for re-entry. If returns to PSH are excluded, for example, the tool will not disqualify any returns due to overlap with PSH enrollments, even if they exist.
        </p>
    </div>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">4. Classifying Return Timing</h3>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="background-color: {NeutralPalette.NEUTRAL_BG};">
                    <th style="padding: 10px; text-align: left; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">Category</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">Days from Exit</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};"><strong>&lt; 6 months</strong></td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">0–180 days</td>
                </tr>
                <tr style="background-color: {NeutralPalette.NEUTRAL_BG_LIGHT};">
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};"><strong>6–12 months</strong></td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">181–365 days</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};"><strong>12–24 months</strong></td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">366–730 days</td>
                </tr>
                <tr style="background-color: {NeutralPalette.NEUTRAL_BG_LIGHT};">
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};"><strong>&gt; 24 months</strong></td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">731+ days</td>
                </tr>
            </tbody>
        </table>
    </div>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">5. Interpretation Guide</h3>
        <ul style="margin-left: 20px;">
            <li style="margin-bottom: 10px;">
                <strong>Return Rate</strong><br>
                <span style="color: {NeutralPalette.NEUTRAL_MUTED};">Percentage of clients who re‑enroll in any homeless service after exiting to Permanent Housing</span>
            </li>
            <li style="margin-bottom: 10px;">
                <strong>Timing Distribution</strong><br>
                <span style="color: {NeutralPalette.NEUTRAL_MUTED};">How long (in days) it takes for clients to return, broken down by the categories above</span>
            </li>
            <li>
                <strong>Trajectory Flows</strong><br>
                <span style="color: {NeutralPalette.NEUTRAL_MUTED};">Sankey or flow diagrams showing client pathways from exit through return (if any)</span>
            </li>
        </ul>
    </div>
</div>
"""

ABOUT_INBOUND_CONTENT = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: {NeutralPalette.NEUTRAL_TEXT}
    ; line-height: 1.6;">
    <p style="margin-bottom: 20px;">
        <strong style="font-size: 1.1em; color: {NeutralPalette.NEUTRAL_HEADER}
    ;">Inbound Recidivism Analysis Overview</strong><br>
        <em style="color: {NeutralPalette.NEUTRAL_MUTED}
    ;">Track each client’s first re-entry into homelessness services following a prior exit within the reporting period.</em>
    </p>

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}
    ; font-size: 1em; margin-bottom: 10px;">Key Features:</h3>
        <ul style="margin-left: 20px;">
            <li style="margin-bottom: 8px;"><strong>Configurable Lookback Window:</strong> Define how many days before the entry date to search for a preceding exit.</li>
            <li style="margin-bottom: 8px;"><strong>First Entry per Client:</strong> Only the earliest <em>ProjectStart</em> between <em>report_start</em> and <em>report_end</em> is analyzed for each client.</li>
            <li style="margin-bottom: 8px;"><strong>Customizable Filters:</strong> Apply separate CoC, local CoC, agency, program, and project-type filters on entries and exits.</li>
            <li style="margin-bottom: 8px;"><strong>Return Classification:</strong>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li>🆕 <strong>New Enrollments:</strong> No exit found within the lookback window.</li>
                    <li>🔄 <strong>Returning:</strong> Most recent exit in the lookback window to a non-permanent-housing destination.</li>
                    <li>🏠 <strong>Returning from Housing:</strong> Most recent exit destination was <code style="background-color: {NeutralPalette.NEUTRAL_BG}
    ; padding: 2px 4px; border-radius: 3px;">Permanent Housing Situations</code>.</li>
                </ul>
            </li>
            <li><strong>Interactive Visuals:</strong> Summary metrics cards, time-to-entry box plots, flow pivot tables, and Sankey diagrams.</li>
        </ul>
    </div>

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}
    ; font-size: 1em; margin-bottom: 10px;">Methodology Highlights:</h3>
        <ol style="margin-left: 20px;">
            <li style="margin-bottom: 8px;"><strong>Entry Selection:</strong> Filter by date range, sort by <em>ProjectStart</em>, then select the first entry per client.</li>
            <li style="margin-bottom: 8px;"><strong>Exit Lookup:</strong> For each selected entry, identify the <strong>single most recent</strong> exit within <em>lookback_days</em> prior to entry.</li>
            <li><strong>Classification Logic:</strong>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li><em>New</em> if no prior exit in the window.</li>
                    <li><em>Returning from Housing</em> if the exit’s <em>ExitDestinationCat</em> equals <code style="background-color: {NeutralPalette.NEUTRAL_BG}
    ; padding: 2px 4px; border-radius: 3px;">Permanent Housing Situations</code>.</li>
                    <li><em>Returning</em> otherwise.</li>
                </ul>
            </li>
        </ol>
    </div>

    <div style="background-color: rgba(0, 102, 204, 0.1); padding: 15px; border-radius: 4px; border-left: 3px solid {NeutralPalette.INFO};">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">Interpretation Guide:</h3>
        <ul style="margin-left: 20px; margin-bottom: 0;">
            <li style="margin-bottom: 8px;"><strong>Summary Metrics:</strong> Counts and percentages for <em>New</em>, <em>Returning</em>, and <em>Returning from Housing</em>.</li>
            <li style="margin-bottom: 8px;"><strong>Timing Analysis:</strong> Box plots display the distribution of days between exit and entry.</li>
            <li><strong>Flow Analysis:</strong> Pivot tables and Sankey diagrams visualize exit → entry pathways.</li>
        </ul>
    </div>
</div>
"""

ABOUT_OUTBOUND_CONTENT = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: {NeutralPalette.NEUTRAL_TEXT}; line-height: 1.6;">
    <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1.1em; margin-bottom: 15px;">How This Analysis Works</h3>

    <div style="margin-bottom: 25px;">
        <h4 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 0.95em; margin-bottom: 10px;">1. Which Exits Are Included?</h4>
        <p style="margin-left: 20px;">
            You configure filters (CoC, Agency, Program, Project Type, Destination Category, etc.) and a reporting date range.<br>
            For each client, only their <strong>last exit</strong> enrollment (matching filters and within the date window) is analyzed.
        </p>
    </div>

    <div style="margin-bottom: 25px;">
        <h4 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 0.95em; margin-bottom: 10px;">2. Return Definitions</h4>
        <ul style="margin-left: 20px;">
            <li style="margin-bottom: 10px;">
                <strong>Return</strong> = first enrollment after the exit (any project type).
            </li>
            <li>
                <strong>Return to Homelessness</strong> = first <strong>qualifying</strong> enrollment after exit, defined as:
                <ol style="margin-left: 20px; margin-top: 8px;">
                    <li style="margin-bottom: 5px;">Must be to a homeless project type: Street Outreach (SO), Emergency Shelter (ES), Transitional Housing (TH), Safe Haven (SH), or Permanent Housing (PH).</li>
                    <li style="margin-bottom: 5px;">Non-PH homeless enrollments (SO, ES, TH, SH) qualify immediately as returns to homelessness.</li>
                    <li>For PH enrollments, additional rules apply:
                        <ul style="margin-left: 20px; margin-top: 5px;">
                            <li><strong>Skip</strong> any PH enrollment where <code style="background-color: {NeutralPalette.NEUTRAL_BG}; padding: 2px 4px; border-radius: 3px;">ProjectStart == HouseholdMoveInDate</code>.</li>
                            <li>If gap ≤ 14 days, build an <strong>exclusion window</strong> (from day +1 after start to exit +14 days).</li>
                            <li>Skip any future PH entry falling within any exclusion window.</li>
                            <li>Otherwise, that PH entry <strong>qualifies</strong> as a return to homelessness.</li>
                        </ul>
                    </li>
                </ol>
            </li>
        </ul>

        <p style="background-color: rgba(217, 119, 6, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid {NeutralPalette.WARNING}; margin-top: 15px;">
            <em><strong>Note:</strong> "Return to Homelessness" rates are computed <strong>only</strong> for exits to Permanent Housing.</em>
        </p>
    </div>

    <div style="margin-bottom: 25px;">
        <h4 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 0.95em; margin-bottom: 10px;">3. Key Output Metrics</h4>
        <ul style="margin-left: 20px;">
            <li style="margin-bottom: 8px;"><strong># Exits Analyzed:</strong> count of last exits matching your filters.</li>
            <li style="margin-bottom: 8px;"><strong># Exits to PH:</strong> count where <code style="background-color: {NeutralPalette.NEUTRAL_BG}; padding: 2px 4px; border-radius: 3px;">ExitDestinationCat == "Permanent Housing Situations"</code>.</li>
            <li style="margin-bottom: 8px;"><strong>Return & % Return:</strong> # and share of exits with any subsequent enrollment.</li>
            <li style="margin-bottom: 8px;"><strong>Return to Homelessness (from PH) & %:</strong> among PH exits, # and share meeting the above "Return to Homelessness" criteria.</li>
            <li><strong>Timing Metrics:</strong> median/average/max days between exit and qualifying return.</li>
        </ul>
    </div>

    <div style="margin-bottom: 25px;">
        <h4 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 0.95em; margin-bottom: 10px;">4. PH vs. Non-PH Comparison</h4>
        <p style="margin-left: 20px;">
            Side-by-side metrics for exits to Permanent Housing vs. all other exits.<br>
            PH-specific "Return to Homelessness" metrics are shown only for the PH subset.
        </p>
    </div>
</div>
"""

ABOUT_GENERAL_ANALYSIS_CONTENT = f"""
<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif; color: {NeutralPalette.NEUTRAL_TEXT}; line-height: 1.6;">
    <p style="margin-bottom: 20px;">
        <strong style="font-size: 1.1em; color: {NeutralPalette.NEUTRAL_HEADER};">General Analysis Methodology</strong><br>
        <em style="color: {NeutralPalette.NEUTRAL_MUTED};">Comprehensive HMIS data analysis with metrics, demographics, trends, length of stay, and equity analysis across your entire dataset.</em>
    </p>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">1. Overview - Core Metric Calculations</h3>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">System Flow Metrics</h4>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li style="margin-bottom: 8px;">
                <strong>Clients Served:</strong> Unique clients with active enrollment anytime during period<br>
                <code style="background-color: {NeutralPalette.NEUTRAL_BG}; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;">
                (ProjectExit ≥ period_start OR ProjectExit is NULL) AND ProjectStart ≤ period_end
                </code>
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Households Served:</strong> Count of heads of household only (IsHeadOfHousehold = "yes")
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Inflow:</strong> Clients entering programs who weren't in ANY programs the day before period started<br>
                <code style="background-color: {NeutralPalette.NEUTRAL_BG}; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;">
                ProjectStart BETWEEN period_start AND period_end AND ClientID NOT IN active_before_period
                </code>
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Outflow:</strong> Clients exiting programs who aren't in ANY programs on the last day of period<br>
                <code style="background-color: {NeutralPalette.NEUTRAL_BG}; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;">
                ProjectExit BETWEEN period_start AND period_end AND ClientID NOT IN active_at_period_end
                </code>
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Net Flow:</strong> Inflow - Outflow (positive = system growth)
            </li>
        </ul>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Housing Outcome Metrics</h4>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li style="margin-bottom: 8px;">
                <strong>PH Exits:</strong> Unique clients who exited to permanent housing destinations<br>
                <code style="background-color: {NeutralPalette.NEUTRAL_BG}; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;">
                ExitDestinationCat = "Permanent Housing Situations" AND ProjectExit BETWEEN period_start AND period_end
                </code>
            </li>
            <li style="margin-bottom: 8px;">
                <strong>PH Exit Rate:</strong> Percentage of unique exiting clients who went to permanent housing<br>
                <code style="background-color: {NeutralPalette.NEUTRAL_BG}; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;">
                (Unique PH exit clients ÷ Unique clients with any exit) × 100
                </code>
            </li>
        </ul>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Returns to Homelessness (HUD-Compliant Logic)</h4>
        <ol style="margin-left: 20px; margin-bottom: 15px;">
            <li style="margin-bottom: 10px;">
                <strong>Base Population:</strong> Clients who exited to permanent housing during reporting period
            </li>
            <li style="margin-bottom: 10px;">
                <strong>Search Window:</strong> From PH exit date + configurable return window (default 180 days)
            </li>
            <li style="margin-bottom: 10px;">
                <strong>Return Identification Rules:</strong>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li style="margin-bottom: 5px;">Any enrollment in SO, ES, TH, or SH = immediate return</li>
                    <li style="margin-bottom: 5px;">For PH re-entries:
                        <ul style="margin-left: 20px; margin-top: 5px;">
                            <li>Skip if ProjectStart = HouseholdMoveInDate (direct placement)</li>
                            <li>Skip if within 14 days of PH exit (short stays create exclusion windows)</li>
                            <li>Skip if within any prior exclusion window</li>
                            <li>Otherwise counts as return to homelessness</li>
                        </ul>
                    </li>
                </ul>
            </li>
        </ol>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Period Comparison Analysis</h4>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li style="margin-bottom: 8px;">
                <strong>Carryover Clients:</strong> Active in both current and previous periods
            </li>
            <li style="margin-bottom: 8px;">
                <strong>New Clients:</strong> Active in current period but not in previous period
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Exited Clients:</strong> Active in previous period but not in current period
            </li>
        </ul>

        <p style="background-color: rgba(217, 119, 6, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid {NeutralPalette.WARNING};">
            <strong>IMPORTANT:</strong> When filters are active, inflow/outflow only track movement within filtered programs. However, returns to homelessness are ALWAYS tracked system-wide regardless of filters.
        </p>
    </div>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">2. Demographics - Breakdown Analysis</h3>

        <p style="margin-bottom: 10px;">Breaks down all core metrics by demographic dimensions:</p>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li>Race/Ethnicity, Gender, Age Tier at Entry</li>
            <li>Program CoC, Local CoC, Agency Name, Program Name</li>
            <li>Project Type, Household Type, Head of Household Status</li>
            <li>Veteran Status, Chronic Homelessness, Currently Fleeing DV</li>
        </ul>

        <p style="margin-bottom: 10px;"><strong>For each demographic group, calculates:</strong></p>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li>Served, Inflow, Outflow, Net Flow</li>
            <li>Total Exits, PH Exits, PH Exit Rate</li>
            <li>Returns Count, Returns to Homelessness Rate</li>
        </ul>

        <p style="background-color: rgba(0, 102, 204, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid {NeutralPalette.INFO};">
            <strong>Note:</strong> Returns use the FULL unfiltered dataset to scan for re-entries, ensuring accurate system-wide tracking even when viewing filtered subsets.
        </p>
    </div>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">3. Trends - Time Series Analysis</h3>

        <p style="margin-bottom: 10px;"><strong>Time Aggregation Options:</strong></p>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li>Days (D), Weeks (W), Months (M), Quarters (Q), Years (Y)</li>
        </ul>

        <p style="margin-bottom: 10px;"><strong>For each time period, recalculates:</strong></p>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li>All core metrics (served, inflow, outflow, PH exits, returns)</li>
            <li>Period-over-period changes (absolute and percentage)</li>
            <li>Rolling averages (configurable window)</li>
            <li>Growth rates by demographic groups</li>
        </ul>

        <p style="background-color: rgba(0, 102, 204, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid {NeutralPalette.INFO};">
            <strong>Note:</strong> Each time bucket is calculated independently using the full metric logic, not simple counts. This ensures accurate deduplication and business rule application.
        </p>
    </div>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">4. Length of Stay Analysis</h3>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Analysis Level</h4>
        <p style="margin-bottom: 10px; background-color: rgba(217, 119, 6, 0.1); padding: 10px; border-radius: 4px; border-left: 3px solid {NeutralPalette.WARNING};">
            <strong>IMPORTANT:</strong> This is ENROLLMENT-LEVEL analysis. Each enrollment is counted separately - a client with multiple enrollments appears multiple times. All days are calculated using entry/exit dates (not bed nights).
        </p>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Calculation Method</h4>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li style="margin-bottom: 8px;">
                <strong>For Exited Enrollments:</strong> Exit Date - Entry Date + 1
            </li>
            <li style="margin-bottom: 8px;">
                <strong>For Active Enrollments:</strong> Report End Date - Entry Date + 1
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Same-Day Services:</strong> Count as 1 day (0 days → 1 day)
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Data Quality Checks:</strong>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li>Exclude records where exit date < entry date</li>
                    <li>Flag stays over 5 years as potential data issues</li>
                    <li>Identify future entry dates</li>
                </ul>
            </li>
        </ul>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Project Type Context</h4>
        <p style="margin-bottom: 10px;">Length of stay interpretation varies by project type.</p>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Additional Features</h4>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li style="margin-bottom: 8px;">
                <strong>Exit Destination Analysis:</strong> Average LOS by exit destination category
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Project Type Analysis:</strong> Compare LOS against HUD benchmarks for each project type
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Context-Aware Recommendations:</strong> Different guidance for emergency vs. permanent housing
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Resource Utilization:</strong> Compare enrollment counts vs. total enrollment days
            </li>
        </ul>
    </div>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">5. Equity Analysis</h3>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Statistical Methods</h4>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li style="margin-bottom: 8px;">
                <strong>Chi-square test:</strong> For groups with n ≥ 5 in all cells (with Yates' correction)
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Fisher's exact test:</strong> For smaller samples
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Significance threshold:</strong> p < 0.05
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Minimum group size:</strong> Configurable (default 30) to ensure statistical reliability
            </li>
        </ul>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Disparity Index Calculation</h4>
        <ul style="margin-left: 20px; margin-bottom: 15px;">
            <li style="margin-bottom: 8px;">
                <strong>For PH Exits (higher is better):</strong><br>
                <code style="background-color: {NeutralPalette.NEUTRAL_BG}; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;">
                DI = Group Rate ÷ Highest Rate
                </code>
            </li>
            <li style="margin-bottom: 8px;">
                <strong>For Returns (lower is better):</strong><br>
                <code style="background-color: {NeutralPalette.NEUTRAL_BG}; padding: 2px 4px; border-radius: 3px; font-size: 0.85em;">
                DI = Lowest Rate ÷ Group Rate (or 1.0 - (Group Rate ÷ Worst Rate) if lowest = 0)
                </code>
            </li>
            <li style="margin-bottom: 8px;">
                <strong>Interpretation:</strong> 1.0 = parity with best group, lower values = larger disparities
            </li>
        </ul>

        <h4 style="color: {NeutralPalette.NEUTRAL_MUTED}; font-size: 0.9em; margin-top: 15px; margin-bottom: 10px;">Disparity Levels</h4>
        <table style="width: 100%; border-collapse: collapse; margin-top: 10px;">
            <thead>
                <tr style="background-color: {NeutralPalette.NEUTRAL_BG};">
                    <th style="padding: 10px; text-align: left; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">Level</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">DI Range</th>
                    <th style="padding: 10px; text-align: left; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">Gap from Best</th>
                </tr>
            </thead>
            <tbody>
                <tr>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER}; color: {NeutralPalette.SUCCESS};">Minimal</td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">0.95–1.0</td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">< 5%</td>
                </tr>
                <tr style="background-color: {NeutralPalette.NEUTRAL_BG_LIGHT};">
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER}; color: {NeutralPalette.INFO};">Moderate</td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">0.80–0.94</td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">5–20%</td>
                </tr>
                <tr>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER}; color: {NeutralPalette.WARNING};">Significant</td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">0.50–0.79</td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">20–50%</td>
                </tr>
                <tr style="background-color: {NeutralPalette.NEUTRAL_BG_LIGHT};">
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER}; color: {NeutralPalette.DANGER};">Severe</td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">< 0.50</td>
                    <td style="padding: 10px; border: 1px solid {NeutralPalette.NEUTRAL_BORDER};">> 50%</td>
                </tr>
            </tbody>
        </table>
    </div>

    <hr style="border: none; border-top: 1px solid {NeutralPalette.NEUTRAL_BORDER}; margin: 20px 0;">

    <div style="margin-bottom: 25px;">
        <h3 style="color: {NeutralPalette.NEUTRAL_HEADER}; font-size: 1em; margin-bottom: 10px;">6. Important Technical Details</h3>

        <ul style="margin-left: 20px;">
            <li style="margin-bottom: 10px;">
                <strong>Client vs. Enrollment Level:</strong>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li>Most metrics: Client-level (each person counted once)</li>
                    <li>Length of Stay: Enrollment-level (each enrollment counted separately)</li>
                </ul>
            </li>
            <li style="margin-bottom: 10px;">
                <strong>Date Inclusivity:</strong> All date ranges are inclusive
            </li>
            <li style="margin-bottom: 10px;">
                <strong>Missing Data Handling:</strong>
                <ul style="margin-left: 20px; margin-top: 5px;">
                    <li>NULL exit dates treated as "still active"</li>
                    <li>Missing demographics shown as "Not Reported"</li>
                    <li>Zero-day stays converted to 1 day</li>
                </ul>
            </li>
            <li style="margin-bottom: 10px;">
                <strong>Filter Application Order:</strong>
                <ol style="margin-left: 20px; margin-top: 5px;">
                    <li>Date range filters applied first</li>
                    <li>Demographic/program filters applied second</li>
                    <li>Metrics calculated on filtered dataset</li>
                    <li>Returns tracked on FULL dataset regardless of filters</li>
                </ol>
        </ul>
    </div>
</div>
"""

# ==================== ENHANCED HELPER FUNCTIONS ====================


def render_about_section(
    title: str, content: str, expanded: bool = False, icon: str = "📘"
):
    """
    Render an about/help section with enhanced styling.
    Content is preserved from original.
    """
    # Add custom styling for expanders
    st.markdown(
        f"""
    <style>
    div[data-testid="stExpander"] {{
        background: {NeutralPalette.BG_SUBTLE};
        border: 1px solid {NeutralPalette.BORDER_SUBTLE};
        border-radius: 8px;
        margin-bottom: 16px;
    }}

    div[data-testid="stExpander"] > div:first-child {{
        padding: 4px 8px;
    }}

    div[data-testid="stExpander"] button {{
        font-weight: 600;
        font-size: 1rem;
    }}

    div[data-testid="stExpander"] button:hover {{
        background: {NeutralPalette.BG_LIGHT};
    }}
    </style>
    """,
        unsafe_allow_html=True,
    )

    with st.expander(f"{icon} {title}", expanded=expanded):
        st.html(content)


def create_info_card(
    title: str, content: str, type: str = "info", icon: Optional[str] = None
) -> str:
    """
    Create an enhanced information card with better visual design.
    """
    # Type configurations with improved styling
    type_config = {
        "info": {
            "bg": "rgba(0, 102, 204, 0.08)",
            "border": f"{NeutralPalette.INFO}",
            "border_bg": "rgba(0, 102, 204, 0.15)",
            "icon": "ℹ️",
        },
        "success": {
            "bg": "rgba(5, 152, 98, 0.08)",
            "border": f"{NeutralPalette.SUCCESS}",
            "border_bg": "rgba(5, 152, 98, 0.15)",
            "icon": "✅",
        },
        "warning": {
            "bg": "rgba(217, 119, 6, 0.08)",
            "border": f"{NeutralPalette.WARNING}",
            "border_bg": "rgba(217, 119, 6, 0.15)",
            "icon": "⚠️",
        },
        "danger": {
            "bg": "rgba(220, 38, 38, 0.08)",
            "border": f"{NeutralPalette.DANGER}",
            "border_bg": "rgba(220, 38, 38, 0.15)",
            "icon": "❌",
        },
    }

    config = type_config.get(type, type_config["info"])
    icon = icon or config["icon"]

    return f"""
    <div style="
        background: {config["bg"]};
        border: 1px solid {config["border_bg"]};
        border-left: 4px solid {config["border"]};
        padding: 20px;
        border-radius: 8px;
        margin: 16px 0;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: all 0.2s ease;
    ">
        <h4 style="
            margin: 0 0 12px 0;
            color: {config["border"]};
            display: flex;
            align-items: center;
            gap: 10px;
            font-size: 1.1rem;
            font-weight: 600;
        ">
            <span style="font-size: 1.3em;">{icon}</span>
            {title}
        </h4>
        <div style="
            color: inherit;
            opacity: {NeutralPalette.TEXT_SECONDARY};
            line-height: 1.6;
        ">
            {content}
        </div>
    </div>
    """


def create_metric_highlight(
    label: str,
    value: str,
    subtext: Optional[str] = None,
    color: Optional[str] = None,
    trend: Optional[str] = None,
) -> str:
    """
    Create an enhanced metric highlight with better visual design.
    """
    color = color or NeutralPalette.PRIMARY
    subtext_html = (
        f"""
        <p style="
            margin: 8px 0 0 0;
            font-size: 0.875rem;
            opacity: {NeutralPalette.TEXT_MUTED};
            line-height: 1.4;
        ">{subtext}</p>
    """
        if subtext
        else ""
    )

    trend_html = (
        f"""
        <div style="
            margin-top: 12px;
            font-size: 0.9rem;
            color: {color};
            font-weight: 500;
        ">{trend}</div>
    """
        if trend
        else ""
    )

    return f"""
    <div style="
        text-align: center;
        padding: 24px;
        background: {NeutralPalette.BG_LIGHT};
        border-radius: 12px;
        border: 1px solid {NeutralPalette.BORDER_SUBTLE};
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.04);
        transition: all 0.2s ease;
        position: relative;
        overflow: hidden;
    ">
        <div style="
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            height: 3px;
            background: {color};
        "></div>
        <div style="
            font-size: 0.875rem;
            opacity: {NeutralPalette.TEXT_MUTED};
            margin-bottom: 12px;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            font-weight: 600;
        ">{label}</div>
        <div style="
            font-size: 2.5rem;
            font-weight: 700;
            color: {color};
            line-height: 1;
            letter-spacing: -0.02em;
        ">{value}</div>
        {subtext_html}
        {trend_html}
    </div>
    """


# ==================== EXPORT ALL PUBLIC FUNCTIONS ====================

__all__ = [
    "NeutralPalette",
    # "render_header",  # Not defined in this file
    "render_footer",
    "render_about_section",
    "create_info_card",
    "create_metric_highlight",
    "get_header_logo_html",
    "get_footer_html",
    "ABOUT_SPM2_CONTENT",
    "ABOUT_INBOUND_CONTENT",
    "ABOUT_OUTBOUND_CONTENT",
    "ABOUT_GENERAL_ANALYSIS_CONTENT",
]

# ==================== LEGACY COMPATIBILITY ====================

# Keep these for backward compatibility
HTML_HEADER_LOGO = get_header_logo_html()
HTML_FOOTER = get_footer_html()
