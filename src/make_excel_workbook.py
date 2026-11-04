"""Build the Excel workbook of CTR pivot tables (excel/ctr_pivot_tables.xlsx).

Run from PyCharm (right-click > Run) or from the repo root, after
ctr_mid_report.py has run once so the sample is cached:
    python src/make_excel_workbook.py

The Data sheet holds every row of the sample (train, validation and test) with
the columns the pivot tables need. Each other sheet is a real Excel pivot table
on that sheet: impressions (count of click), clicks (sum of click) and CTR
(average of click) by hour, day, site category, app category, device type,
banner position and connection type, plus two CTR grids (site category by
device type, hour by day). Every pivot table has a "split" filter set to
train, so the numbers match outputs/ctr_by_*.csv. Pick (All) in the filter to
see the whole sample instead.

openpyxl can't make pivot tables itself, so this script writes the sheets with
openpyxl and then adds the pivot cache and pivot table XML parts to the file.
The cells under each pivot table are filled in with the same numbers, so the
workbook reads correctly even before Excel refreshes it (it refreshes on open).
"""

import re
import shutil
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape

import pandas as pd
from openpyxl import Workbook
from openpyxl.cell import WriteOnlyCell
from openpyxl.chart import BarChart, LineChart, Reference
from openpyxl.formatting.rule import ColorScaleRule
from openpyxl.styles import Font

from ctr_mid_report import (CACHE_DIR, ROOT, SAMPLE_FRAC, SEED,
                            add_time_features, log, split)

XLSX = ROOT / "excel" / "ctr_pivot_tables.xlsx"
DAY_NAMES = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FILTER_FIELD, FILTER_VALUE = "split", "train"

# Data sheet columns, in order. "num" columns are numbers, the rest text.
COLUMNS = [("split", "text"), ("date", "text"), ("day_of_week", "text"),
           ("hour_of_day", "num"), ("site_category", "text"),
           ("app_category", "text"), ("device_type", "num"),
           ("banner_pos", "num"), ("device_conn_type", "num"), ("click", "num")]
COL_NAMES = [c for c, _ in COLUMNS]
CLICK = COL_NAMES.index("click")

# One pivot table per sheet. by_impressions sorts rows by impressions, most first.
PIVOTS = [
    dict(sheet="CTR by hour", rows="hour_of_day", label="Hour of day",
         title="CTR by hour of day", chart="line"),
    dict(sheet="CTR by day", rows="day_of_week", label="Day of week",
         title="CTR by day of week", chart="bar"),
    dict(sheet="CTR by site category", rows="site_category", label="Site category",
         title="CTR by site category (Avazu hashes the category names)",
         chart="bar", by_impressions=True),
    dict(sheet="CTR by app category", rows="app_category", label="App category",
         title="CTR by app category (Avazu hashes the category names)",
         chart="bar", by_impressions=True),
    dict(sheet="CTR by device type", rows="device_type", label="Device type",
         title="CTR by device type", chart="bar"),
    dict(sheet="CTR by banner position", rows="banner_pos", label="Banner position",
         title="CTR by banner position", chart="bar"),
    dict(sheet="CTR by connection type", rows="device_conn_type",
         label="Connection type", title="CTR by device connection type", chart="bar"),
    dict(sheet="Site x device", rows="site_category", cols="device_type",
         label="Site category", col_label="Device type",
         title="CTR by site category and device type (the baseline's main groups)"),
    dict(sheet="Hour x day", rows="hour_of_day", cols="day_of_week",
         label="Hour of day", col_label="Day of week",
         title="CTR by hour of day and day of week"),
]
MEASURES = [("Impressions", "count", 3, "#,##0"), ("Clicks", "sum", 3, "#,##0"),
            ("CTR", "average", 10, "0.00%")]   # name, Excel function, numFmtId, format
TOP = 5  # pivot tables start on row 5; the split filter sits on row 3

NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"
NS_R = "http://schemas.openxmlformats.org/officeDocument/2006/relationships"
REL = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/"
CT = "application/vnd.openxmlformats-officedocument.spreadsheetml."


# ---------------------------------------------------------------- data
def load_data():
    cache = CACHE_DIR / f"sample_frac{SAMPLE_FRAC}_seed{SEED}.pkl"
    if not cache.exists():
        raise FileNotFoundError(f"{cache} not found. Run ctr_mid_report.py first.")
    df = add_time_features(pd.read_pickle(cache))
    train, val, test = split(df)
    parts = []
    for name, part in (("train", train), ("validation", val), ("test", test)):
        ts = pd.to_datetime(part["hour"], format="%y%m%d%H")
        dow = ts.dt.dayofweek
        parts.append(pd.DataFrame({
            "split": name,
            "date": ts.dt.strftime("%Y-%m-%d"),
            "day_of_week": dow.astype(str) + " " + dow.map(dict(enumerate(DAY_NAMES))),
            "hour_of_day": ts.dt.hour.astype(int),
            "site_category": part["site_category"].astype(str),
            "app_category": part["app_category"].astype(str),
            "device_type": part["device_type"].astype(int),
            "banner_pos": part["banner_pos"].astype(int),
            "device_conn_type": part["device_conn_type"].astype(int),
            "click": part["click"].astype(int),
        }))
    data = pd.concat(parts, ignore_index=True)
    # rows in time order, as in the raw file
    data = data.sort_values(["date", "hour_of_day"], kind="stable").reset_index(drop=True)
    return data


# ---------------------------------------------------------------- pivot layout
def shared_items(data):
    """Sorted distinct values of every column except click (the pivot cache's item lists)."""
    return {c: sorted(data[c].unique().tolist()) for c in COL_NAMES if c != "click"}


def layout(p, data, items):
    """Work out what the pivot table shows: its rows, columns and cell values."""
    train = data[data[FILTER_FIELD] == FILTER_VALUE]
    if "cols" in p:
        t = train.pivot_table(index=p["rows"], columns=p["cols"], values="click",
                              aggfunc="mean")
        row_vals = sorted(t.index.tolist())
        col_vals = sorted(t.columns.tolist())
        t = t.loc[row_vals, col_vals]
        t["Grand Total"] = train.groupby(p["rows"])["click"].mean()
        total = train.groupby(p["cols"])["click"].mean().reindex(col_vals).tolist()
        total.append(train["click"].mean())
        return dict(row_vals=row_vals, col_vals=col_vals, values=t.values.tolist(),
                    total=total)
    g = train.groupby(p["rows"])["click"].agg(["count", "sum", "mean"])
    if p.get("by_impressions"):
        # ties keep the shared-item order, which is how Excel breaks them too
        order = sorted(g.index, key=lambda v: (-g.loc[v, "count"], items[p["rows"]].index(v)))
    else:
        order = sorted(g.index)
    g = g.loc[order]
    total = [len(train), int(train["click"].sum()), train["click"].mean()]
    return dict(row_vals=order, values=g.values.tolist(), total=total)


# ---------------------------------------------------------------- sheets
def bold(ws, value, size=None):
    c = WriteOnlyCell(ws, value)
    c.font = Font(bold=True, size=size) if size else Font(bold=True)
    return c


def fmt(ws, value, number_format):
    c = WriteOnlyCell(ws, None if pd.isna(value) else value)
    c.number_format = number_format
    return c


def write_about(wb, data):
    ws = wb.create_sheet("About")
    ws.column_dimensions["A"].width = 110
    counts = data[FILTER_FIELD].value_counts()
    lines = [
        ("Click-through rate pivot tables: Avazu mobile ad sample", 14),
        (None, None),
        (f"Data sheet: {len(data):,} ad impressions, the {SAMPLE_FRAC:.2%} random sample of "
         "the Avazu training file (Oct 21-30 2014, seed 42) that ctr_mid_report.py uses.", None),
        (f"split column: train {counts.get('train', 0):,} rows, validation "
         f"{counts.get('validation', 0):,}, test {counts.get('test', 0):,} "
         "(test is the held-out last day, Oct 30).", None),
        ("click is 1 if the ad was clicked, else 0. Each pivot table counts it (impressions), "
         "sums it (clicks) and averages it (CTR = clicks / impressions).", None),
        (None, None),
        (f"Every pivot table is filtered to split = {FILTER_VALUE}, so its numbers match "
         "outputs/ctr_by_*.csv and the data the baseline learns from.", None),
        ("Change the split filter (cell B3 on each sheet) to (All) or test to see other rows. "
         "Right-click a pivot table > Refresh if Excel hasn't refreshed it.", None),
        ("Site and app categories are hashed in the Avazu data, so they show as codes like "
         "50e219e0, not names.", None),
        ("day_of_week: 0 = Monday. hour_of_day is the hour (UTC) the ad was shown.", None),
        (None, None),
        ("Sheets:", None),
    ]
    lines += [(f"  {p['sheet']}: {p['title']}", None) for p in PIVOTS]
    lines += [(None, None),
              ("Built by src/make_excel_workbook.py from the cached sample. "
               "Rerun it after changing the sample.", None)]
    for text, size in lines:
        ws.append([bold(ws, text, size) if size else text])


def write_data(wb, data):
    ws = wb.create_sheet("Data")
    ws.freeze_panes = "A2"
    for i, w in enumerate([11, 11, 12, 12, 14, 14, 12, 11, 17, 7]):
        ws.column_dimensions[chr(65 + i)].width = w
    ws.append([bold(ws, c) for c in COL_NAMES])
    for row in data.itertuples(index=False, name=None):
        ws.append(row)
    return f"A1:{chr(64 + len(COL_NAMES))}{len(data) + 1}"


def write_pivot_sheet(wb, p, lay):
    """Write the title, the filter cells and the pivot table's current values.
    Returns the pivot table's cell range."""
    ws = wb.create_sheet(p["sheet"])
    ws.append([bold(ws, p["title"], 13)])
    ws.append(["Impressions = number of ads shown, Clicks = number clicked, "
               "CTR = Clicks / Impressions." if "cols" not in p else
               "Each cell is the CTR (clicks / impressions) for that combination."])
    ws.append([bold(ws, FILTER_FIELD), FILTER_VALUE])
    ws.append([])
    if "cols" in p:
        ncols = len(lay["col_vals"]) + 2
        ws.append([bold(ws, "CTR"), bold(ws, p["col_label"])])
        ws.append([bold(ws, p["label"])] + [bold(ws, v) for v in lay["col_vals"]]
                  + [bold(ws, "Grand Total")])
        for r, vals in zip(lay["row_vals"], lay["values"]):
            ws.append([r] + [fmt(ws, v, "0.00%") for v in vals])
        ws.append([bold(ws, "Grand Total")] + [fmt(ws, v, "0.00%") for v in lay["total"]])
        nrows = len(lay["row_vals"]) + 3
        ws.column_dimensions["A"].width = 16
        for i in range(1, ncols):
            ws.column_dimensions[chr(65 + i)].width = 10
        last_col = chr(64 + ncols)
        ws.conditional_formatting.add(
            f"B{TOP + 2}:{chr(64 + ncols - 1)}{TOP + 1 + len(lay['row_vals'])}",
            ColorScaleRule(start_type="min", start_color="FFF7FBFF",
                           end_type="max", end_color="FF2171B5"))
    else:
        ncols = 1 + len(MEASURES)
        ws.append([bold(ws, p["label"])] + [bold(ws, m[0]) for m in MEASURES])
        for r, vals in zip(lay["row_vals"], lay["values"]):
            ws.append([r] + [fmt(ws, v, m[3]) for v, m in zip(vals, MEASURES)])
        ws.append([bold(ws, "Grand Total")]
                  + [fmt(ws, v, m[3]) for v, m in zip(lay["total"], MEASURES)])
        nrows = len(lay["row_vals"]) + 2
        ws.column_dimensions["A"].width = 18
        for col in "BCD":
            ws.column_dimensions[col].width = 13
        last_col = chr(64 + ncols)
        add_chart(ws, p, len(lay["row_vals"]))
    return f"A{TOP}:{last_col}{TOP + nrows - 1}"


def add_chart(ws, p, n):
    """Chart of the CTR column under the pivot table (rows TOP+1 .. TOP+n)."""
    chart = LineChart() if p["chart"] == "line" else BarChart()
    if p["chart"] == "bar":
        chart.type = "col"
    chart.title = p["title"].split(" (")[0] + " (training days)"
    chart.y_axis.title = "CTR"
    chart.y_axis.number_format = "0%"
    chart.x_axis.title = p["label"]
    chart.y_axis.delete = False
    chart.x_axis.delete = False
    chart.legend = None
    chart.add_data(Reference(ws, min_col=4, min_row=TOP, max_row=TOP + n), titles_from_data=True)
    chart.set_categories(Reference(ws, min_col=1, min_row=TOP + 1, max_row=TOP + n))
    chart.width, chart.height = 22, 11
    ws.add_chart(chart, "F5")


# ---------------------------------------------------------------- pivot XML
def xml_value(v):
    return f'<n v="{v}"/>' if isinstance(v, (int, float)) else f'<s v="{escape(str(v))}"/>'


def cache_definition_xml(data, items, ref):
    fields = []
    for c in COL_NAMES:
        if c == "click":
            fields.append('<cacheField name="click" numFmtId="0"><sharedItems '
                          'containsSemiMixedTypes="0" containsString="0" containsNumber="1" '
                          'containsInteger="1" minValue="0" maxValue="1"/></cacheField>')
            continue
        vals = items[c]
        if isinstance(vals[0], int):
            attrs = (f'containsSemiMixedTypes="0" containsString="0" containsNumber="1" '
                     f'containsInteger="1" minValue="{min(vals)}" maxValue="{max(vals)}" ')
        else:
            attrs = ""
        fields.append(f'<cacheField name="{c}" numFmtId="0"><sharedItems {attrs}'
                      f'count="{len(vals)}">{"".join(xml_value(v) for v in vals)}'
                      f'</sharedItems></cacheField>')
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<pivotCacheDefinition xmlns="{NS}" xmlns:r="{NS_R}" r:id="rId1" '
            f'refreshOnLoad="1" createdVersion="6" refreshedVersion="6" '
            f'minRefreshableVersion="3" recordCount="{len(data)}">'
            f'<cacheSource type="worksheet"><worksheetSource ref="{ref}" sheet="Data"/>'
            f'</cacheSource><cacheFields count="{len(COL_NAMES)}">{"".join(fields)}'
            f'</cacheFields></pivotCacheDefinition>')


def cache_records_xml(data, items):
    codes = []
    for c in COL_NAMES:
        if c == "click":
            codes.append(data[c].map(lambda v: f'<n v="{v}"/>').tolist())
        else:
            index = {v: f'<x v="{i}"/>' for i, v in enumerate(items[c])}
            codes.append(data[c].map(index).tolist())
    body = "".join("<r>" + "".join(row) + "</r>" for row in zip(*codes))
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<pivotCacheRecords xmlns="{NS}" xmlns:r="{NS_R}" count="{len(data)}">'
            f'{body}</pivotCacheRecords>')


def pivot_table_xml(p, lay, items, ref, name):
    rows, cols = p["rows"], p.get("cols")
    data_by_value = {v: i for i, v in enumerate(items[rows])}
    fields = []
    for c in COL_NAMES:
        if c == "click":
            fields.append('<pivotField dataField="1" showAll="0"/>')
            continue
        n = len(items[c])
        item_xml = (f'<items count="{n + 1}">' + "".join(f'<item x="{i}"/>' for i in range(n))
                    + '<item t="default"/></items>')
        if c == FILTER_FIELD:
            fields.append(f'<pivotField axis="axisPage" showAll="0">{item_xml}</pivotField>')
        elif c == rows:
            sort = ""
            scope = ""
            if p.get("by_impressions"):
                sort = ' sortType="descending"'
                scope = ('<autoSortScope><pivotArea dataOnly="0" outline="0" fieldPosition="0">'
                         '<references count="1"><reference field="4294967294" count="1" '
                         'selected="0"><x v="0"/></reference></references></pivotArea>'
                         '</autoSortScope>')
            fields.append(f'<pivotField axis="axisRow" showAll="0"{sort}>{item_xml}{scope}'
                          f'</pivotField>')
        elif c == cols:
            fields.append(f'<pivotField axis="axisCol" showAll="0">{item_xml}</pivotField>')
        else:
            fields.append('<pivotField showAll="0"/>')

    def x(i):
        return "<x/>" if i == 0 else f'<x v="{i}"/>'

    row_items = "".join(f"<i>{x(data_by_value[v])}</i>" for v in lay["row_vals"])
    row_items += '<i t="grand"><x/></i>'
    n_row_items = len(lay["row_vals"]) + 1
    row_fields = f'<rowFields count="1"><field x="{COL_NAMES.index(rows)}"/></rowFields>'
    page = (f'<pageFields count="1"><pageField fld="{COL_NAMES.index(FILTER_FIELD)}" '
            f'item="{items[FILTER_FIELD].index(FILTER_VALUE)}" hier="-1"/></pageFields>')
    if cols:
        col_index = {v: i for i, v in enumerate(items[cols])}
        col_fields = f'<colFields count="1"><field x="{COL_NAMES.index(cols)}"/></colFields>'
        col_items = "".join(f"<i>{x(col_index[v])}</i>" for v in lay["col_vals"])
        col_items += '<i t="grand"><x/></i>'
        n_col_items = len(lay["col_vals"]) + 1
        data_fields = (f'<dataFields count="1"><dataField name="CTR" fld="{CLICK}" '
                       f'subtotal="average" baseField="0" baseItem="0" numFmtId="10"/>'
                       f'</dataFields>')
        loc = f'firstHeaderRow="1" firstDataRow="2" firstDataCol="1"'
        captions = (f' rowHeaderCaption="{escape(p["label"])}" '
                    f'colHeaderCaption="{escape(p["col_label"])}"')
    else:
        col_fields = '<colFields count="1"><field x="-2"/></colFields>'
        col_items = "".join(f'<i i="{k}">{x(k)}</i>' if k else "<i><x/></i>"
                            for k in range(len(MEASURES)))
        n_col_items = len(MEASURES)
        data_fields = (f'<dataFields count="{len(MEASURES)}">'
                       + "".join(f'<dataField name="{m}" fld="{CLICK}" subtotal="{f}" '
                                 f'baseField="0" baseItem="0" numFmtId="{nf}"/>'
                                 for m, f, nf, _ in MEASURES)
                       + "</dataFields>")
        loc = f'firstHeaderRow="0" firstDataRow="1" firstDataCol="1"'
        captions = f' rowHeaderCaption="{escape(p["label"])}"'
    return (f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<pivotTableDefinition xmlns="{NS}" name="{escape(name)}" cacheId="1" '
            f'applyNumberFormats="0" applyBorderFormats="0" applyFontFormats="0" '
            f'applyPatternFormats="0" applyAlignmentFormats="0" applyWidthHeightFormats="1" '
            f'dataCaption="Values" updatedVersion="6" minRefreshableVersion="3" '
            f'useAutoFormatting="1" itemPrintTitles="1" createdVersion="6" indent="0" '
            f'outline="1" outlineData="1" multipleFieldFilters="0"{captions}>'
            f'<location ref="{ref}" {loc} rowPageCount="1" colPageCount="1"/>'
            f'<pivotFields count="{len(COL_NAMES)}">{"".join(fields)}</pivotFields>'
            f'{row_fields}<rowItems count="{n_row_items}">{row_items}</rowItems>'
            f'{col_fields}<colItems count="{n_col_items}">{col_items}</colItems>'
            f'{page}{data_fields}'
            f'<pivotTableStyleInfo name="PivotStyleMedium2" showRowHeaders="1" '
            f'showColHeaders="1" showRowStripes="0" showColStripes="0" showLastColumn="1"/>'
            f'</pivotTableDefinition>')


def add_pivot_parts(path, data, items, data_ref, pivot_refs, layouts):
    """Add the pivot cache and one pivot table per pivot sheet to the saved workbook."""
    tmp = path.with_suffix(".tmp.xlsx")
    with zipfile.ZipFile(path) as zin:
        files = {n: zin.read(n) for n in zin.namelist()}

    wb_xml = files["xl/workbook.xml"].decode()
    sheet_names = re.findall(r'<sheet [^>]*name="([^"]+)"', wb_xml)

    files["xl/pivotCache/pivotCacheDefinition1.xml"] = cache_definition_xml(
        data, items, data_ref).encode()
    files["xl/pivotCache/pivotCacheRecords1.xml"] = cache_records_xml(data, items).encode()
    files["xl/pivotCache/_rels/pivotCacheDefinition1.xml.rels"] = (
        f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        f'<Relationship Id="rId1" Type="{REL}pivotCacheRecords" '
        f'Target="pivotCacheRecords1.xml"/></Relationships>').encode()

    # workbook: register the cache
    wb_rels = files["xl/_rels/workbook.xml.rels"].decode()
    wb_rels = wb_rels.replace("</Relationships>",
                              f'<Relationship Id="rIdPivotCache1" Type="{REL}pivotCacheDefinition" '
                              f'Target="pivotCache/pivotCacheDefinition1.xml"/></Relationships>')
    files["xl/_rels/workbook.xml.rels"] = wb_rels.encode()
    caches = f'<pivotCaches><pivotCache cacheId="1" r:id="rIdPivotCache1"/></pivotCaches>'
    if "<calcPr" in wb_xml:
        wb_xml = re.sub(r"(<calcPr[^>]*/>)", r"\1" + caches, wb_xml, count=1)
    else:
        wb_xml = wb_xml.replace("</workbook>", caches + "</workbook>")
    if 'xmlns:r=' not in wb_xml.split(">", 2)[1]:
        wb_xml = wb_xml.replace("<workbook ", f'<workbook xmlns:r="{NS_R}" ', 1)
    files["xl/workbook.xml"] = wb_xml.encode()

    types = files["[Content_Types].xml"].decode()
    overrides = [(f"/xl/pivotCache/pivotCacheDefinition1.xml", "pivotCacheDefinition"),
                 (f"/xl/pivotCache/pivotCacheRecords1.xml", "pivotCacheRecords")]

    for k, p in enumerate(PIVOTS, 1):
        sheet_no = sheet_names.index(p["sheet"]) + 1
        name = "pt " + p["sheet"]
        files[f"xl/pivotTables/pivotTable{k}.xml"] = pivot_table_xml(
            p, layouts[k - 1], items, pivot_refs[k - 1], name).encode()
        files[f"xl/pivotTables/_rels/pivotTable{k}.xml.rels"] = (
            f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
            f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
            f'<Relationship Id="rId1" Type="{REL}pivotCacheDefinition" '
            f'Target="../pivotCache/pivotCacheDefinition1.xml"/></Relationships>').encode()
        rels_path = f"xl/worksheets/_rels/sheet{sheet_no}.xml.rels"
        rel = (f'<Relationship Id="rIdPivot{k}" Type="{REL}pivotTable" '
               f'Target="../pivotTables/pivotTable{k}.xml"/>')
        if rels_path in files:
            files[rels_path] = files[rels_path].decode().replace(
                "</Relationships>", rel + "</Relationships>").encode()
        else:
            files[rels_path] = (
                f'<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
                f'{rel}</Relationships>').encode()
        overrides.append((f"/xl/pivotTables/pivotTable{k}.xml", "pivotTable"))

    types = types.replace("</Types>", "".join(
        f'<Override PartName="{part}" ContentType="{CT}{kind}+xml"/>'
        for part, kind in overrides) + "</Types>")
    files["[Content_Types].xml"] = types.encode()

    with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
        # content types first, as Excel writes it
        zout.writestr("[Content_Types].xml", files.pop("[Content_Types].xml"))
        for n, b in files.items():
            zout.writestr(n, b)
    shutil.move(tmp, path)


# ---------------------------------------------------------------- main
def main():
    log("Loading the cached sample")
    data = load_data()
    items = shared_items(data)
    log(f"  {len(data):,} rows; " + ", ".join(
        f"{s} {n:,}" for s, n in data[FILTER_FIELD].value_counts().items()))

    layouts = [layout(p, data, items) for p in PIVOTS]

    log("Writing the sheets")
    XLSX.parent.mkdir(exist_ok=True)
    wb = Workbook(write_only=True)
    write_about(wb, data)
    pivot_refs = [write_pivot_sheet(wb, p, lay) for p, lay in zip(PIVOTS, layouts)]
    data_ref = write_data(wb, data)
    wb.save(XLSX)

    log("Adding the pivot tables")
    add_pivot_parts(XLSX, data, items, data_ref, pivot_refs, layouts)
    log(f"Saved {XLSX.relative_to(ROOT)} ({XLSX.stat().st_size / 1e6:.1f} MB)")


if __name__ == "__main__":
    main()
