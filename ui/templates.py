"""
HTML templates and logo components
"""

import streamlit as st

# Header logo HTML
HTML_HEADER_LOGO = """
<div style="background-color: #111111; padding: 12px 20px; border-radius: 8px; text-align: left;">
    <a href="https://icalliances.org/" target="_blank">
        <img src="https://images.squarespace-cdn.com/content/v1/54ca7491e4b000c4d5583d9c/eb7da336-e61c-4e0b-bbb5-1a7b9d45bff6/Dash+Logo+2.png?format=1000w" width="350">
    </a>
</div>
"""

# Footer HTML with copyright and logos
HTML_FOOTER = """
<div style="background-color: #111111; padding: 30px 25px; border-radius: 12px; margin-top: 50px;">
    <div style="font-style: italic; color: #aaaaaa; text-align: center; font-size: 18px;">
        <a href="https://icalliances.org/" target="_blank">
            <img src="https://images.squarespace-cdn.com/content/v1/54ca7491e4b000c4d5583d9c/eb7da336-e61c-4e0b-bbb5-1a7b9d45bff6/Dash+Logo+2.png?format=900w" width="160"> 
        </a>
        <div style="margin-top: 10px;">DASH™ is a trademark of Institute for Community Alliances.</div>
    </div>
    <div style="font-style: italic; color: #aaaaaa; text-align: center; margin-top: 20px; font-size: 16px;">
        <a href="https://icalliances.org/" target="_blank">
            <img src="https://images.squarespace-cdn.com/content/v1/54ca7491e4b000c4d5583d9c/1475614371395-KFTYP42QLJN0VD5V9VB1/ICA+Official+Logo+PNG+%28transparent%29.png?format=1500w" width="120">
        </a>
        <div style="margin-top: 8px;">© 2025 Institute for Community Alliances (ICA). All rights reserved.</div>
    </div>
</div>
"""

# About section content
ABOUT_SPM2_CONTENT = """
**SPM2 Analysis Overview**  
*Assessing housing stability by tracking if and when clients return to homeless services after exiting to permanent housing.*

---

### 1. Client Universe  
- **Included Project Types**  
- Street Outreach (SO)  
- Emergency Shelter (ES)  
- Transitional Housing (TH)  
- Safe Haven (SH)  
- Permanent Housing (PH)  
- **Date Window**  
- Consider exits that occurred **within 730 days before** the reporting period  
    - Exit Date ≥ (Report Start Date – 730 days)  
    - Exit Date ≤ (Report End Date – 730 days)

---

### 2. Identifying the "Permanent Housing" Exit  
- **Definition:** the client's **earliest** exit into any PH project  
- **Tie‑Breaker:** if multiple exits share that date, pick the one with **lowest Enrollment ID**

---

### 3. Scanning for a Return to Homeless Services  
1. **Search Window:** from the PH exit date up to the end of the reporting period  
2. **Eligible Return Enrollments:**  
- Any SO, ES, TH, or SH project  
- PH project re‑entries that meet both:  
    - Start date **> 14 days** after the original PH exit  
    - No overlap with **another PH transition window**  
    - Transition window = Day 1 after PH start through `min(PH exit + 14 days, report end)`  
3. **First Qualifying Return:**  
- Project Start ≥ PH Exit Date  
- Project Start ≤ Report End Date  
- Stop at the very first enrollment that fulfills the above

---

### 4. Classifying Return Timing  
| Category         | Days from Exit       |  
|------------------|----------------------|  
| **< 6 months**   | 0–180 days           |  
| **6–12 months**  | 181–365 days         |  
| **12–24 months** | 366–730 days         |  
| **> 24 months**  | 731+ days            |

---

### 5. Interpretation Guide  
- **Return Rate**  
- Percentage of clients who re‑enroll in any homeless service after exiting to PH  
- **Timing Distribution**  
- How long (in days) it takes for clients to return, broken down by the categories above  
- **Trajectory Flows**  
- Sankey or flow diagrams showing client pathways from exit through return (if any)
"""

ABOUT_INBOUND_CONTENT = """
**Inbound Recidivism Analysis Overview**  
*Evaluate client returns to homelessness programs after prior exits.*  

### Key Features:
- **Configurable Lookback Window:** Choose how many days before entry to search for exits.
- **First-Entry Selection:** Takes each client's first entry between *report_start* and *report_end*.
- **Flexible Filters:** Apply CoC, local CoC, agency, program, and project-type filters separately on entries and exits.
- **Return Classification:**
  - 🆕 **New Clients:** No exit found within the lookback window.
  - 🔄 **Returning Clients:** Last exit within lookback, but not to permanent housing.
  - 🏠 **Returning from Housing:** Last exit destination was `Permanent Housing Situations`.
- **Interactive Visuals:** Metrics cards, time‑to‑entry box plots, flow matrices, and Sankey diagrams.

### Methodology Highlights:
1. **Entry Identification:** Filtered by date range, then deduplicated to first entry per client.  
2. **Exit Lookup:** For each entry, find the **single most recent** exit within *lookback_days*.  
3. **Classification Logic:**  
   - *New* if no qualifying exit  
   - *Returning From Housing* if exit's destination == `Permanent Housing Situations`  
   - *Returning* otherwise  

**Interpretation Guide:**
- **Metrics:** Counts & percentages for *New*, *Returning*, and *Returning from Housing*.  
- **Timing Analysis:** Box plots show distribution of days between exit and entry.  
- **Flow Analysis:** Matrices & Sankey diagrams visualize exit→entry paths.  
"""

ABOUT_OUTBOUND_CONTENT = """
### How This Analysis Works

1. **Which Exits Are Included?**  
   You configure filters (CoC, Agency, Program, Project Type, Destination Category, etc.) and a reporting date range.  
   For each client, only their **last exit** enrollment (matching filters and within the date window) is analyzed.

2. **Return Definitions**  
   - **Return** = first enrollment after the exit (any project type).  
   - **Return to Homelessness** = first **qualifying** enrollment after exit, defined as:  
     1. Exclude any project where `ProjectTypeCode` is a non-homeless project type.
     2. **Skip** any PH enrollment where `ProjectStart == HouseholdMoveInDate`.  
     3. For PH enrollments:  
        - If gap ≤ 14 days, build an **exclusion window** (from day +1 after start to exit +14 days).  
        - Also skip any future PH entry falling within any exclusion window.  
        - Otherwise, that PH entry **qualifies** as a return from Permanent Housing.
     4. Any non-PH enrollment (after those filters) qualifies immediately.  

   *Note:* "Return to Homelessness" rates are computed **only** for exits to Permanent Housing.

3. **Key Output Metrics**  
   - **# Exits Analyzed:** count of last exits matching your filters.  
   - **# Exits to PH:** count where `ExitDestinationCat == "Permanent Housing Situations"`.  
   - **Return & % Return:** # and share of exits with any subsequent enrollment.  
   - **Return to Homelessness (PH) & %:** among PH exits, # and share meeting the above "Return to Homelessness" criteria.  
   - **Timing Metrics:** median/average/max days between exit and qualifying return.

4. **PH vs. Non-PH Comparison**  
   Side-by-side metrics for exits to Permanent Housing vs. all other exits.  
   PH-specific "Return to Homelessness" metrics are shown only for the PH subset.
"""

def render_header():
    """Render the application header with logo"""
    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""
        - Upload your data file to get started.
        - Use the **Reset Session** button if you need to start over.
        - Navigate between the available analyses:
            1. **SPM 2 Analysis**
            2. **Inbound Recidivism Analysis**
            3. **Outbound Recidivism Analysis**
            4. **General Analysis**
        """)
    with col2:
        st.markdown(HTML_HEADER_LOGO, unsafe_allow_html=True)

def render_footer():
    """Render the application footer"""
    st.markdown(HTML_FOOTER, unsafe_allow_html=True)

def render_about_section(title: str, content: str, expanded: bool = False):
    """
    Render an about/help section with consistent styling.
    
    Parameters:
        title (str): Section title
        content (str): Markdown content
        expanded (bool): Whether the section is expanded by default
    """
    with st.expander(f"📘 {title}", expanded=expanded):
        st.markdown(content)