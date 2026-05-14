# UI Layout & Visual Polish Audit — SPM2 Streamlit App

## Context

The previous plan covered three broad ROI improvements (column_config, dark theme, fragments) but mixed component additions with layout work. This plan replaces it with a **pure layout/spacing/proportion/alignment audit** under the hard constraints you set:

- No widgets, charts, tables, buttons, or inputs are added, removed, or replaced.
- No analysis logic, data, or outputs change.
- No widget behavior or interactivity changes.
- Only containers, columns, ratios, spacing, dividers, grouping, and visual hierarchy are in scope.

**CSS scope** (your choice): Scoped CSS is allowed when a real visual issue can't be solved in pure primitives, and any new CSS lands in `src/ui/themes/styles.py` — no inline CSS scattered across page files.

**Surfaces audited** (your choice): `main.py` + sidebar + module routing; Dashboard module + its 5 sub-tabs; SPM2 page; Recidivism (inbound + outbound).

**Context7 / Streamlit docs verified for current primitives** (Streamlit ≥1.57.0, project pin):
- `st.columns(spec, *, gap, vertical_alignment, border, width)` — `gap` accepts `xxsmall|xsmall|small|medium|large|xlarge|xxlarge|None`; `vertical_alignment` accepts `top|center|bottom`.
- `st.container(*, border, gap, horizontal, horizontal_alignment, height, width)` — `horizontal=True` arranges children in a row that wraps on overflow; `horizontal_alignment` accepts `left|center|right|distribute`.
- `st.divider()`, `st.tabs()`, `st.expander()` — unchanged from prior versions.

These are the only layout primitives this plan relies on, plus the existing `html_factory` and `ui` factories already in the codebase.

---

## Prioritized findings

Order: **highest visual impact first; lowest risk first within ties**. Risk is "low" everywhere because no component is being modified — only its container or arrangement. Each finding declares the fix type: **PRIMITIVE** (Streamlit only) or **CENTRALIZED CSS** (lands in `src/ui/themes/styles.py`).

### Tier A — High visual impact

#### A1. Sibling pages diverge on the info-box API (SPM2 vs Inbound/Outbound)
- **Where:** SPM2 uses `html_factory.info_box(...)` at [src/modules/spm2/page.py:168-171](src/modules/spm2/page.py#L168-L171), [src/modules/spm2/page.py:617-622](src/modules/spm2/page.py#L617-L622), [src/modules/spm2/page.py:770-775](src/modules/spm2/page.py#L770-L775), [src/modules/spm2/page.py:862-867](src/modules/spm2/page.py#L862-L867). Inbound and Outbound use `ui.info_section(...)` at [src/modules/recidivism/inbound_page.py:519-524](src/modules/recidivism/inbound_page.py#L519-L524), [src/modules/recidivism/inbound_page.py:738-743](src/modules/recidivism/inbound_page.py#L738-L743), [src/modules/recidivism/inbound_page.py:826-831](src/modules/recidivism/inbound_page.py#L826-L831), [src/modules/recidivism/outbound_page.py:536-541](src/modules/recidivism/outbound_page.py#L536-L541), [src/modules/recidivism/outbound_page.py:789-794](src/modules/recidivism/outbound_page.py#L789-L794), [src/modules/recidivism/outbound_page.py:917-922](src/modules/recidivism/outbound_page.py#L917-L922).
- **What looks off:** Hierarchy/grouping. Three conceptually-sibling pages render the same kind of contextual note with two visually distinct components — different padding, border weight, and color treatments. Even if individually fine, they break sibling cohesion.
- **Proposed change:** Pick one API and apply it across all three pages (no component is added; the *other* one already exists at each call site as an alternative styling). Recommendation: keep `ui.info_section` since it's used 2:1 and supports expanders, then make `html_factory.info_box` render to the same DOM by reusing the same CSS class in `src/ui/factories/html.py`'s template. The class lives in `src/ui/themes/styles.py`.
- **Fix type:** PRIMITIVE for call-site swaps; CENTRALIZED CSS if you'd rather keep both helpers and merge their visual treatment in `styles.py`.
- **Risk:** Low — call signature matches at each site; behavior is identical.

#### A2. Demographics summary metric row is too dense
- **Where:** [src/modules/dashboard/demographics.py:1546-1551](src/modules/dashboard/demographics.py#L1546-L1551) — `summary_cols = st.columns(5)` holding five `st.metric` calls.
- **What looks off:** Proportion. Five equal columns means ~240px each on a 1200px viewport, ~190px on the 1024px tablet width. Labels truncate, values feel cramped.
- **Proposed change:** Replace `st.columns(5)` with `st.container(horizontal=True, gap="small")`. Per the docs, `horizontal=True` arranges children in a row sized to content and wraps cleanly when the viewport is narrow. Visual rhythm holds on desktop; it stops looking broken on smaller windows.
- **Fix type:** PRIMITIVE.
- **Risk:** Low — `st.metric` accepts being placed inside a horizontal container without any signature change.

#### A3. Styled dataframes have a different visual language than adjacent charts
- **Where:** Demographics breakdown table at [src/modules/dashboard/demographics.py:1724-1728](src/modules/dashboard/demographics.py#L1724-L1728); equity disparity table at [src/modules/dashboard/equity.py:1543-1557](src/modules/dashboard/equity.py#L1543-L1557); LOS table at [src/modules/dashboard/length_of_stay.py:1309](src/modules/dashboard/length_of_stay.py#L1309).
- **What looks off:** Grouping/hierarchy. Plotly charts above sit inside the implicit chart-card frame from your theme; the styled dataframes drop straight onto the page background with no matching frame, breaking the visual rhythm of "card → card → card".
- **Proposed change:** Wrap each `st.dataframe(...)` call in a `with st.container(border=True): ...` block. No data is touched; only a border frame is added. This is the primitive that the `building-streamlit-dashboards` skill explicitly recommends for "visual card separations".
- **Fix type:** PRIMITIVE.
- **Risk:** Low — wrapping a `st.dataframe` in a bordered container has no effect on column behavior, sorting, or scroll.

#### A4. SPM2 lookback radio is visually detached from the number it controls
- **Where:** [src/modules/spm2/page.py:109-164](src/modules/spm2/page.py#L109-L164) — `st.radio` for "Days vs Months" sits above a `st.number_input` for the lookback value, with an `html_factory.divider()` between them.
- **What looks off:** Grouping. The radio's *only* purpose is to set the unit of the number below it. The divider implies they're separate concepts. They aren't.
- **Proposed change:** Wrap both inside one `st.container(gap="small")` and remove the intermediate divider. Keep both widgets exactly as they are. If the sidebar feels too tight, use `gap="xsmall"` per the docs.
- **Fix type:** PRIMITIVE.
- **Risk:** Low — divider removal is the only change; radio and number_input keep all their kwargs and `key=` values.

### Tier B — Medium visual impact

#### B1. Selectbox pairs across all three sibling pages use a 50:50 column split with no breathing room
- **Where:** SPM2 flow analysis: [src/modules/spm2/page.py:778](src/modules/spm2/page.py#L778) and [src/modules/spm2/page.py:870-879](src/modules/spm2/page.py#L870-L879). Inbound: [src/modules/recidivism/inbound_page.py:745](src/modules/recidivism/inbound_page.py#L745) and [src/modules/recidivism/inbound_page.py:833](src/modules/recidivism/inbound_page.py#L833). Outbound: [src/modules/recidivism/outbound_page.py:796](src/modules/recidivism/outbound_page.py#L796) and [src/modules/recidivism/outbound_page.py:924](src/modules/recidivism/outbound_page.py#L924).
- **What looks off:** Spacing. The labels above each selectbox are long ("Exit Dimension: Rows" / "Entry Dimension: Columns"). With `gap="small"` (the default), the labels visually crowd into one another.
- **Proposed change:** Change `st.columns(2)` → `st.columns(2, gap="medium", vertical_alignment="bottom")`. The `vertical_alignment="bottom"` aligns the selectbox baselines when the two labels wrap to a different number of lines (which they will, on narrower viewports).
- **Fix type:** PRIMITIVE.
- **Risk:** Low — only column kwargs change; selectboxes are unchanged.

#### B2. Focus-filter selectboxes are not visually distinct from the dimension selectboxes above them
- **Where:** The second selectbox pair on each flow tab: [src/modules/spm2/page.py:870-879](src/modules/spm2/page.py#L870-L879), [src/modules/recidivism/inbound_page.py:833-845](src/modules/recidivism/inbound_page.py#L833-L845), [src/modules/recidivism/outbound_page.py:924-935](src/modules/recidivism/outbound_page.py#L924-L935).
- **What looks off:** Hierarchy. Two different conceptual jobs (choose dimensions vs. focus the network) look identical. Users have to read the warning info-box to know they're different.
- **Proposed change:** Wrap each focus-filter pair in `with st.container(border=True): ...`. The existing `:material/...` icon or short caption above already labels them; the border provides the missing visual separation.
- **Fix type:** PRIMITIVE.
- **Risk:** Low.

#### B3. Sidebar "Analysis Type" title is too heavy in a narrow sidebar
- **Where:** [main.py:527-566](main.py#L527-L566) — uses `html_factory.title(level=4)` for "Analysis Type" plus an inline HTML module info card immediately below.
- **What looks off:** Hierarchy. The `level=4` title has padding, a left border, and a background tint (per `html_factory`), which is heavy for a narrow ~280-350px sidebar slot.
- **Proposed change:** Either (a) drop to `level=5` in this one location, or (b) wrap the title + selectbox + info card in one `st.container(gap="small")` so the eye reads it as a single block rather than three stacked elements with their own visual weight.
- **Fix type:** PRIMITIVE.
- **Risk:** Low — `html_factory.title` accepts a `level` kwarg; no rendering pathway changes.

#### B4. Dashboard summary period display uses an unbalanced 50:50 split
- **Where:** [src/modules/dashboard/summary.py:603-620](src/modules/dashboard/summary.py#L603-L620) — `col1, col2 = st.columns(2)` for the reporting/lookback period info boxes.
- **What looks off:** Proportion. Two different-length info boxes share equal width; the shorter one looks padded and the longer one looks cramped.
- **Proposed change:** Use `st.columns([3, 2], gap="medium")` (or `[1, 1]` is OK but with `gap="medium"` to space the cards). Picks up the same rhythm as the SPM2 selectbox-pair fix (B1).
- **Fix type:** PRIMITIVE.
- **Risk:** Low.

#### B5. Filter sub-groups inside expanders stack flat
- **Where:** [src/modules/spm2/page.py:236-466](src/modules/spm2/page.py#L236-L466) (Exit Filters: 8 multiselects; Return Filters: 6 multiselects), and corresponding sections in `inbound_page.py` and `outbound_page.py`. Filter rendering driven by [src/modules/dashboard/filters.py](src/modules/dashboard/filters.py).
- **What looks off:** Grouping. Inside one expander, 6–8 multiselects stack with no visual sub-grouping. Logically related ones (CoC + LocalCoC, Program + Project Type, etc.) get the same visual weight as unrelated ones.
- **Proposed change:** Within each expander, insert one or two `html_factory.divider()` calls between conceptual sub-groups (e.g., after the "geography" group, after the "program" group). No nested columns — the sidebar is too narrow for 2-column filter rows and would look worse.
- **Fix type:** PRIMITIVE.
- **Risk:** Low — only dividers added, no widgets touched.

### Tier C — Low visual impact (polish)

#### C1. Sidebar divider placement is conditional and inconsistent
- **Where:** [main.py:405-459](main.py#L405-L459) — `st.divider()` is only emitted when `has_data` is true; the rest of the sidebar uses a mix of `st.html(html_factory.divider())` and bare `st.markdown("---")`.
- **What looks off:** Spacing/consistency.
- **Proposed change:** Standardize on `html_factory.divider()` for all sidebar separators, and emit unconditionally between the data-status block, the action-button row, and the session-management block.
- **Fix type:** PRIMITIVE.
- **Risk:** Low.

#### C2. Dialog headings and dividers don't match the rest of the app
- **Where:** Export/import dialogs at [main.py:676](main.py#L676), [main.py:706](main.py#L706), [main.py:726](main.py#L726), [main.py:770](main.py#L770), [main.py:805](main.py#L805), [main.py:824](main.py#L824), [main.py:922](main.py#L922) — `st.markdown("### …")` for headings and `st.markdown("---")` for dividers.
- **What looks off:** Hierarchy. The dialogs look like a different app inside the same app.
- **Proposed change:** Swap `st.markdown("### …")` → `st.html(html_factory.title(..., level=3))` and `st.markdown("---")` → `st.html(html_factory.divider())`. No content change.
- **Fix type:** PRIMITIVE.
- **Risk:** Low.

#### C3. Welcome screen 2×2 module card grid lacks explicit gap
- **Where:** [main.py:270](main.py#L270) — `st.columns(2)` with `min-height: 200px` cards inside.
- **What looks off:** Proportion. Cards are forced to equal height by the `min-height`; with the default `gap="small"`, the two columns sit close together.
- **Proposed change:** `st.columns(2, gap="medium")`.
- **Fix type:** PRIMITIVE.
- **Risk:** Low.

#### C4. Duplicate-data warning floats loose at the top of module pages
- **Where:** [main.py:1006-1019](main.py#L1006-L1019) — `show_duplicate_info()` followed by a `st.divider()`.
- **What looks off:** Grouping. The warning is logically part of the data-summary block but is rendered as a free-floating note.
- **Proposed change:** Wrap the call + divider in `with st.container(border=True): ...` so it reads as a single attention block.
- **Fix type:** PRIMITIVE.
- **Risk:** Low.

#### C5. Dashboard tab styling uses brittle negative margins
- **Where:** [src/modules/dashboard/page.py:47-259](src/modules/dashboard/page.py#L47-L259) — CSS `margin: 0 -5rem; padding: 1rem 5rem;` on `.stTabs [data-baseweb="tab-list"]`.
- **What looks off:** This is a CSS hack to bleed the tab strip past the page padding. It works but is brittle against Streamlit version updates that change the DOM structure.
- **Proposed change:** Move the rule into `src/ui/themes/styles.py` (where the rest of the app's centralized CSS lives) and simplify to `margin: 0; padding: 1rem 0;`. Loses the full-bleed effect — confirm that's acceptable before changing.
- **Fix type:** CENTRALIZED CSS.
- **Risk:** Low; reversible.

---

## Accepted constraints (will not fix)

These are real visual irritants that pure Streamlit (or our scoped-CSS policy) can't cleanly fix. Calling them out so the plan is honest:

- **Equity disparity table has 9 columns of mixed numeric types** ([src/modules/dashboard/equity.py:1510](src/modules/dashboard/equity.py#L1510)). Pure-layout options are bad: splitting into two side-by-side `st.dataframe` calls counts as adding a component; horizontal scroll is worse than the status quo. The real fix is `column_config` (a display-formatting feature on the existing widget) — that was item #1 of the previous plan and is **not** included here per your "no component changes" constraint.
- **Streamlit does not allow precise vertical-rhythm control** between e.g. a `st.metric` row and the chart that follows. The gap between them is set by Streamlit's internal element margin. We can't shrink it without injecting CSS that targets Streamlit's internal `[data-testid="stVerticalBlock"]` (fragile and explicitly out of scope here).
- **Sibling pages diverge on date-config density** (SPM2 has 4 inputs in the sidebar block, Inbound has 2, Outbound has 1). This is *justified* — SPM2 needs the lookback unit + return-period inputs that Inbound and Outbound don't. Not a layout bug; not changing.
- **Filter sub-grouping into two-column rows within sidebar expanders** would look cramped at typical sidebar widths. Stick with dividers (B5), not columns.

---

## Implementation pass strategy

The findings cluster into three reasonably-independent commits:

1. **Sibling-page consistency pass** (A1 + B1 + B2 + B5) — touches SPM2 + inbound + outbound page files and `src/ui/themes/styles.py`. Highest perceived improvement because three pages start looking like a family.
2. **Dashboard density & framing pass** (A2 + A3 + B4 + C1 + C2 + C4) — touches dashboard module files and `main.py`. Pure primitives, mostly mechanical.
3. **Sidebar & polish pass** (A4 + B3 + C3 + C5) — touches `main.py`, SPM2 page sidebar block, and `src/ui/themes/styles.py`.

Order doesn't matter mechanically; you can also pick a subset.

## Critical files

- [main.py](main.py)
- [src/modules/spm2/page.py](src/modules/spm2/page.py)
- [src/modules/recidivism/inbound_page.py](src/modules/recidivism/inbound_page.py)
- [src/modules/recidivism/outbound_page.py](src/modules/recidivism/outbound_page.py)
- [src/modules/dashboard/page.py](src/modules/dashboard/page.py), [summary.py](src/modules/dashboard/summary.py), [demographics.py](src/modules/dashboard/demographics.py), [equity.py](src/modules/dashboard/equity.py), [length_of_stay.py](src/modules/dashboard/length_of_stay.py), [filters.py](src/modules/dashboard/filters.py)
- [src/ui/factories/html.py](src/ui/factories/html.py) (for A1's info-box unification, if you keep both helpers)
- [src/ui/themes/styles.py](src/ui/themes/styles.py) (centralized CSS landing zone)

## Verification

No test suite (per CLAUDE.md). Manual verification:

```bash
streamlit run main.py
```

For each commit:

1. Load a sample HMIS CSV and walk every audited surface at three viewport widths: 1920px, 1366px (typical laptop), and 1100px (narrow window).
2. **A1**: confirm SPM2, Inbound, Outbound info boxes are visually indistinguishable on the same browser.
3. **A2 / B4**: shrink the window — the metric / period rows wrap cleanly rather than truncating.
4. **A3 / C4**: confirm tables now sit inside the same card frame as adjacent charts.
5. **A4 / B3**: confirm SPM2 lookback radio + number_input read as one grouped unit and the sidebar's "Analysis Type" block feels right-sized.
6. **B1 / B2 / B5**: flow-analysis sections — verify selectbox pairs breathe, focus filters are visibly framed off, and filter expanders have visual sub-grouping.
7. **C1 / C2 / C3**: open every dialog and the welcome screen — all dividers and headings use the factory.
8. **C5**: confirm tabs still render correctly after the CSS simplification and that no horizontal scroll appears.
9. Toggle the theme switcher (light ↔ dark) on each surface and confirm no new contrast or alignment regressions.
