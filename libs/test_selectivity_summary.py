from __future__ import annotations

import csv
import html
import math
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Iterable
import xml.etree.ElementTree as ET


INPUT_XLSX = Path("data/data_electronic_electrostatic_lumo_regression.xlsx")
OUTPUT_ROOT = Path("data/test")

GROUPS = ("a", "b", "c", "d", "e", "f")
ENTRY_ORDER = ("1", "2", "3", "4", "13", "14", "23", "24", "31", "32", "41", "42")
TEMPERATURE_BY_GROUP = {
    "a": 273.15,
    "b": 298.15,
    "c": 298.15,
    "d": 298.15,
    "e": 298.15,
    "f": 298.15,
}
COMPONENT_COLUMNS = (
    ("electronic_cont", "electronic"),
    ("electrostatic_cont", "electrostatic"),
    ("lumo_cont", "lumo"),
)
PRODUCT_ORDER = ("1-3", "1-4", "2-3", "2-4")
COMPONENT_META = (
    ("electronic", "Electronic", "#8B2F8F"),
    ("electrostatic", "Electrostatic", "#2E8B57"),
    ("lumo", "LUMO", "#8B5A2B"),
)
INITIAL_LABEL_ORDER = ("1", "2", "3", "4")
PATHWAY_TO_PRODUCT = {
    "13": "1-3",
    "31": "1-3",
    "14": "1-4",
    "41": "1-4",
    "23": "2-3",
    "32": "2-3",
    "24": "2-4",
    "42": "2-4",
}
@dataclass(frozen=True)
class EntryRecord:
    entry: str
    name: str
    intercept: float
    prediction: float
    electronic: float
    electrostatic: float
    lumo: float

    @property
    def stage(self) -> str:
        return "initial" if self.entry[1:] in {"1", "2", "3", "4"} else "follow_up"

    @property
    def total_component(self) -> float:
        return self.electronic + self.electrostatic + self.lumo


def base_substrate_name(name: str) -> str:
    """Remove the trailing pathway annotation while keeping internal parentheses."""
    return re.sub(r"\s+\([^()]*\)$", "", name).strip()


def _column_to_index(column_name: str) -> int:
    value = 0
    for character in column_name:
        value = value * 26 + (ord(character.upper()) - 64)
    return value - 1


def _read_cell_text(
    cell: ET.Element,
    shared_strings: list[str],
    namespace: dict[str, str],
) -> str:
    cell_type = cell.get("t")
    value_node = cell.find("a:v", namespace)
    inline_string = cell.find("a:is", namespace)

    if value_node is not None:
        raw_value = value_node.text or ""
        if cell_type == "s":
            return shared_strings[int(raw_value)]
        if cell_type == "b":
            return "TRUE" if raw_value == "1" else "FALSE"
        return raw_value

    if inline_string is not None:
        parts = [
            text_node.text or ""
            for text_node in inline_string.iter(
                "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
            )
        ]
        return "".join(parts)

    return ""


def read_xlsx_rows(path: Path, sheet: str = "xl/worksheets/sheet1.xml") -> list[list[str]]:
    namespace = {"a": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}

    with zipfile.ZipFile(path) as archive:
        shared_strings: list[str] = []
        if "xl/sharedStrings.xml" in archive.namelist():
            shared_root = ET.fromstring(archive.read("xl/sharedStrings.xml"))
            for shared_item in shared_root.findall("a:si", namespace):
                parts = [
                    text_node.text or ""
                    for text_node in shared_item.iter(
                        "{http://schemas.openxmlformats.org/spreadsheetml/2006/main}t"
                    )
                ]
                shared_strings.append("".join(parts))

        worksheet_root = ET.fromstring(archive.read(sheet))
        rows: list[list[str]] = []

        for row in worksheet_root.findall(".//a:sheetData/a:row", namespace):
            cells: dict[int, str] = {}
            max_index = -1

            for cell in row.findall("a:c", namespace):
                reference = cell.get("r", "")
                column_name = "".join(character for character in reference if character.isalpha())
                if not column_name:
                    continue
                index = _column_to_index(column_name)
                max_index = max(max_index, index)
                cells[index] = _read_cell_text(cell, shared_strings, namespace)

            if max_index >= 0:
                rows.append([cells.get(index, "") for index in range(max_index + 1)])

    return rows


def _to_float(raw_value: str) -> float:
    if raw_value in {"", "NAN", "NaN", "nan"}:
        return math.nan
    return float(raw_value)


def load_test_records(path: Path) -> dict[str, list[EntryRecord]]:
    rows = read_xlsx_rows(path)
    header = rows[0]
    index = {name: idx for idx, name in enumerate(header)}
    groups: dict[str, list[EntryRecord]] = {group: [] for group in GROUPS}

    required_columns = {
        "entry",
        "name",
        "intercept",
        "prediction",
        "electronic_cont",
        "electrostatic_cont",
        "lumo_cont",
    }
    missing = required_columns.difference(index)
    if missing:
        raise ValueError(f"Missing columns in regression xlsx: {sorted(missing)}")

    for raw_row in rows[1:]:
        row = raw_row + [""] * (len(header) - len(raw_row))
        entry = row[index["entry"]]
        if not re.fullmatch(r"[a-f]\d+", entry):
            continue

        record = EntryRecord(
            entry=entry,
            name=row[index["name"]],
            intercept=_to_float(row[index["intercept"]]),
            prediction=_to_float(row[index["prediction"]]),
            electronic=_to_float(row[index["electronic_cont"]]),
            electrostatic=_to_float(row[index["electrostatic_cont"]]),
            lumo=_to_float(row[index["lumo_cont"]]),
        )
        groups[entry[0]].append(record)

    for group, records in groups.items():
        groups[group] = sorted(records, key=lambda record: ENTRY_ORDER.index(record.entry[1:]))
        if len(groups[group]) != len(ENTRY_ORDER):
            raise ValueError(
                f"Expected {len(ENTRY_ORDER)} entries for group {group}, found {len(groups[group])}"
            )

    return groups


def _component_values(records: Iterable[EntryRecord], component: str) -> list[float]:
    return [getattr(record, component) for record in records]


def compute_discrimination(records: list[EntryRecord]) -> dict[str, dict[str, float]]:
    scores: dict[str, float] = {}
    spreads: dict[str, float] = {}

    for _, component in COMPONENT_COLUMNS:
        values = _component_values(records, component)
        center = mean(values)
        centered = [value - center for value in values]
        scores[component] = sum(abs(value) for value in centered)
        spreads[component] = math.sqrt(sum(value * value for value in centered) / len(centered))

    total_score = sum(scores.values())
    return {
        component: {
            "center_mean": mean(_component_values(records, component)),
            "discrimination_score": scores[component],
            "share": (scores[component] / total_score) if total_score else 0.0,
            "spread": spreads[component],
        }
        for _, component in COMPONENT_COLUMNS
    }


def compute_pair_advantage(winner: EntryRecord, runner_up: EntryRecord) -> dict[str, float]:
    deltas = {
        component: getattr(runner_up, component) - getattr(winner, component)
        for _, component in COMPONENT_COLUMNS
    }
    total_abs = sum(abs(delta) for delta in deltas.values())
    result: dict[str, float] = {
        "barrier_advantage": runner_up.prediction - winner.prediction,
    }

    for component, delta in deltas.items():
        result[f"{component}_advantage"] = delta
        result[f"{component}_share"] = (abs(delta) / total_abs) if total_abs else 0.0

    return result


def compute_initial_relative_rates(
    records: list[EntryRecord],
    temperature: float,
) -> dict[str, dict[str, float]]:
    gas_constant = 1.987e-3
    raw_rates = {
        record.entry: math.exp(-record.prediction / (gas_constant * temperature))
        for record in records
    }
    total_rate = sum(raw_rates.values())
    return {
        entry: {
            "fraction": (rate / total_rate) if total_rate else 0.0,
            "percent": (rate / total_rate * 100) if total_rate else 0.0,
        }
        for entry, rate in raw_rates.items()
    }


def _logspace(start_power: float, stop_power: float, num: int) -> list[float]:
    if num <= 1:
        return [10 ** start_power]
    step = (stop_power - start_power) / (num - 1)
    return [10 ** (start_power + step * index) for index in range(num)]


def simulate_selectivity(records: list[EntryRecord], temperature: float) -> dict[str, object]:
    entries = {record.entry: record for record in records}
    ordered_predictions = [
        entries[f"{records[0].entry[0]}{suffix}"].prediction for suffix in ENTRY_ORDER
    ]

    k_boltzmann = 1.380649e-23
    planck = 6.62607015e-34
    gas_constant = 1.987e-3

    def rate_constant(delta_g: float) -> float:
        return (k_boltzmann * temperature / planck) * math.exp(
            -delta_g / (gas_constant * temperature)
        )

    (
        k1,
        k2,
        k3,
        k4,
        k13p,
        k14p,
        k23p,
        k24p,
        k31p,
        k32p,
        k41p,
        k42p,
    ) = [rate_constant(delta_g) for delta_g in ordered_predictions]

    k1p_sum = k13p + k14p
    k2p_sum = k23p + k24p
    k3p_sum = k31p + k32p
    k4p_sum = k41p + k42p
    k_a = k1 + k2 + k3 + k4
    max_rate = max(
        k1,
        k2,
        k3,
        k4,
        k13p,
        k14p,
        k23p,
        k24p,
        k31p,
        k32p,
        k41p,
        k42p,
    )
    time_points = [value / max_rate for value in _logspace(-10, 10, 1000)]

    def concentration_of_intermediate(k_i: float, k_ip_sum: float, time_value: float) -> float:
        return (
            k_i
            / (k_ip_sum - k_a)
            * (math.exp(-k_a * time_value) - math.exp(-k_ip_sum * time_value))
        )

    def total_product(k_i: float, k_ijp: float, k_ip_sum: float, time_value: float) -> float:
        term_1 = (1 - math.exp(-k_a * time_value)) / k_a
        term_2 = (1 - math.exp(-k_ip_sum * time_value)) / k_ip_sum
        return k_i * k_ijp / (k_ip_sum - k_a) * (term_1 - term_2)

    p1 = [concentration_of_intermediate(k1, k1p_sum, time_value) for time_value in time_points]
    p2 = [concentration_of_intermediate(k2, k2p_sum, time_value) for time_value in time_points]
    p3 = [concentration_of_intermediate(k3, k3p_sum, time_value) for time_value in time_points]
    p4 = [concentration_of_intermediate(k4, k4p_sum, time_value) for time_value in time_points]
    intermediate_total = [sum(values) for values in zip(p1, p2, p3, p4)]
    max_index = max(range(len(intermediate_total)), key=lambda idx: intermediate_total[idx])

    p13 = [
        total_product(k1, k13p, k1p_sum, time_value) + total_product(k3, k31p, k3p_sum, time_value)
        for time_value in time_points
    ]
    p14 = [
        total_product(k1, k14p, k1p_sum, time_value) + total_product(k4, k41p, k4p_sum, time_value)
        for time_value in time_points
    ]
    p23 = [
        total_product(k2, k23p, k2p_sum, time_value) + total_product(k3, k32p, k3p_sum, time_value)
        for time_value in time_points
    ]
    p24 = [
        total_product(k2, k24p, k2p_sum, time_value) + total_product(k4, k42p, k4p_sum, time_value)
        for time_value in time_points
    ]

    intermediate_labels = ("1", "2", "3", "4")
    intermediate_values = [p1[max_index], p2[max_index], p3[max_index], p4[max_index]]
    intermediate_sum = sum(intermediate_values)
    final_values = [p13[-1], p14[-1], p23[-1], p24[-1]]
    final_sum = sum(final_values)

    return {
        "t_max": time_points[max_index],
        "max_intermediate_total": intermediate_sum,
        "intermediates": [
            {
                "label": label,
                "fraction": (value / intermediate_sum) if intermediate_sum else 0.0,
                "absolute": value,
            }
            for label, value in sorted(
                zip(intermediate_labels, intermediate_values),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
        "final_products": [
            {
                "label": label,
                "fraction": (value / final_sum) if final_sum else 0.0,
                "absolute": value,
            }
            for label, value in sorted(
                zip(PRODUCT_ORDER, final_values),
                key=lambda item: item[1],
                reverse=True,
            )
        ],
    }


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_csv_rows(path: Path, rows: list[list[str]], encoding: str = "utf-8-sig") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding=encoding) as handle:
        writer = csv.writer(handle)
        writer.writerows(rows)


def format_value(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def format_scientific(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}e}"


def product_label_from_entry(entry: str) -> str:
    return PATHWAY_TO_PRODUCT.get(entry[1:], "")


def format_template_measurement(value: float) -> str:
    rounded = f"{value:.2f}"
    if rounded in {"0.00", "-0.00"}:
        return "0.00"
    return rounded.rstrip("0").rstrip(".")


def format_template_rate(value: float) -> str:
    rounded = round(value, 2)
    if math.isclose(rounded, 0.0, abs_tol=0.005):
        return "0.00"
    return f"{rounded:.2f}"


def load_integrated_summary_table(path: Path) -> tuple[list[dict[str, object]], dict[str, dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.reader(handle))

    pathway_rows: list[dict[str, object]] = []
    totals: dict[str, dict[str, str]] = {}

    for row in rows[2:]:
        entry = row[0].strip()
        if not entry:
            continue
        if entry in {"monoalcohol_total", "dialcohol_total"}:
            totals[entry] = {
                "max_point": row[6],
                "final_point": row[7],
            }
            continue

        pathway_rows.append(
            {
                "entry": entry,
                "name": row[1],
                "prediction": row[2],
                "prediction_value": float(row[2]),
                "electronic": row[3],
                "electronic_value": float(row[3]),
                "electrostatic": row[4],
                "electrostatic_value": float(row[4]),
                "lumo": row[5],
                "lumo_value": float(row[5]),
                "max_point": row[6],
                "max_point_value": float(row[6]),
                "final_point": row[7],
                "final_point_value": float(row[7]),
            }
        )

    return pathway_rows, totals


def _escape_svg(text: object) -> str:
    return html.escape(str(text), quote=True)


def _nice_number(value: float) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def _nice_ticks(min_value: float, max_value: float, target_ticks: int = 7) -> list[float]:
    if math.isclose(min_value, max_value):
        min_value -= 1.0
        max_value += 1.0

    span = max_value - min_value
    step = _nice_number(span / max(target_ticks - 1, 1))
    start = math.floor(min_value / step) * step
    end = math.ceil(max_value / step) * step

    ticks: list[float] = []
    current = start
    while current <= end + step * 0.5:
        ticks.append(round(current, 10))
        current += step
    return ticks


def _component_value(record: EntryRecord, component: str) -> float:
    return getattr(record, component)


def build_contribution_svg(
    group: str,
    records: list[EntryRecord],
    initial_metrics: dict[str, dict[str, float]],
    follow_up_metrics: dict[str, dict[str, float]],
    overall_metrics: dict[str, dict[str, float]],
) -> str:
    width = 1480
    height = 930
    plot_left = 250
    plot_top = 140
    plot_width = 760
    row_height = 34
    plot_height = row_height * len(records)
    plot_bottom = plot_top + plot_height

    value_column_x = {
        "electronic": 1085,
        "electrostatic": 1175,
        "lumo": 1265,
        "sum": 1360,
    }

    extents: list[float] = [0.0]
    for record in records:
        negative_extent = 0.0
        positive_extent = 0.0
        for component, _, _ in COMPONENT_META:
            value = _component_value(record, component)
            if value >= 0:
                positive_extent += value
            else:
                negative_extent += value
        extents.extend([negative_extent, positive_extent, record.total_component])

    min_value = min(extents)
    max_value = max(extents)
    span = max_value - min_value
    margin = 0.10 * span if span else 1.0
    ticks = _nice_ticks(min_value - margin, max_value + margin)
    x_min = ticks[0]
    x_max = ticks[-1]

    def scale_x(value: float) -> float:
        return plot_left + (value - x_min) / (x_max - x_min) * plot_width

    svg: list[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="white"/>',
        '<style>',
        'text { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif; fill: #1f2937; }',
        '.title { font-size: 26px; font-weight: 700; }',
        '.subtitle { font-size: 15px; fill: #475569; }',
        '.section { font-size: 18px; font-weight: 700; }',
        '.axis { font-size: 12px; fill: #475569; }',
        '.label { font-size: 12px; }',
        '.small { font-size: 11px; fill: #64748b; }',
        '.value { font-size: 12px; font-weight: 600; }',
        '</style>',
        f'<text class="title" x="32" y="40">Test set {group}: stacked contribution breakdown</text>',
        (
            f'<text class="subtitle" x="32" y="66">'
            f'{_escape_svg(base_substrate_name(records[0].name))}'
            f' | bars are stacked by sign and the black marker shows '
            f'Electronic + Electrostatic + LUMO'
            f'</text>'
        ),
        '<text class="section" x="32" y="108">Pathway-wise stacked decomposition [kcal/mol]</text>',
    ]

    legend_x = 610
    for idx, (_, label, color) in enumerate(COMPONENT_META):
        x = legend_x + idx * 145
        svg.append(f'<rect x="{x}" y="90" width="18" height="18" rx="4" fill="{color}" opacity="0.86"/>')
        svg.append(f'<text class="label" x="{x + 26}" y="104">{_escape_svg(label)}</text>')
    svg.append('<circle cx="1060" cy="99" r="5" fill="#111827"/>')
    svg.append('<text class="label" x="1076" y="104">Total sum</text>')

    svg.append(
        f'<rect x="{plot_left - 165}" y="{plot_top - 8}" width="{plot_width + 540}" '
        f'height="{plot_height + 18}" fill="#fafafa" stroke="#e5e7eb"/>'
    )
    svg.append(
        f'<rect x="{plot_left - 165}" y="{plot_top - 8}" width="{plot_width + 540}" '
        f'height="{row_height * 4}" fill="#f8fafc"/>'
    )
    svg.append(
        f'<text class="small" x="{plot_left - 165}" y="{plot_top - 18}">'
        f'Initial step ({group}1-{group}4)</text>'
    )
    svg.append(
        f'<text class="small" x="{plot_left - 165}" y="{plot_top + row_height * 4 + 8}">'
        f'Follow-up step ({group}13-{group}42)</text>'
    )

    for tick in ticks:
        x = scale_x(tick)
        line_color = "#111827" if math.isclose(tick, 0.0, abs_tol=1e-12) else "#d1d5db"
        line_width = 1.6 if math.isclose(tick, 0.0, abs_tol=1e-12) else 1.0
        svg.append(
            f'<line x1="{x:.2f}" y1="{plot_top - 8}" x2="{x:.2f}" y2="{plot_bottom + 8}" '
            f'stroke="{line_color}" stroke-width="{line_width}"/>'
        )
        svg.append(
            f'<text class="axis" text-anchor="middle" x="{x:.2f}" y="{plot_bottom + 28}">'
            f'{tick:g}</text>'
        )

    svg.append(
        f'<text class="axis" x="{plot_left + plot_width / 2:.2f}" y="{plot_bottom + 52}" '
        f'text-anchor="middle">Stacked contribution value [kcal/mol]</text>'
    )

    for header_key, header_text in (
        ("electronic", "Elec"),
        ("electrostatic", "ES"),
        ("lumo", "LUMO"),
        ("sum", "Sum"),
    ):
        svg.append(
            f'<text class="label" text-anchor="middle" x="{value_column_x[header_key]}" '
            f'y="{plot_top - 18}">{header_text}</text>'
        )

    svg.append(
        f'<line x1="{plot_left + plot_width + 20}" y1="{plot_top - 8}" '
        f'x2="{plot_left + plot_width + 20}" y2="{plot_bottom + 8}" stroke="#d1d5db"/>'
    )

    for row_index, record in enumerate(records):
        y_center = plot_top + row_height * row_index + row_height / 2
        svg.append(
            f'<text class="label" x="{plot_left - 16}" y="{y_center + 4:.2f}" text-anchor="end">'
            f'{_escape_svg(record.entry)}</text>'
        )

        if row_index == 4:
            svg.append(
                f'<line x1="{plot_left - 165}" y1="{y_center - row_height / 2:.2f}" '
                f'x2="{plot_left + plot_width + 370}" y2="{y_center - row_height / 2:.2f}" '
                f'stroke="#cbd5e1" stroke-width="1.2"/>'
            )

        x_zero = scale_x(0.0)
        negative_cursor = 0.0
        positive_cursor = 0.0
        bar_y = y_center - 11
        bar_height = 22

        for component, _, color in COMPONENT_META:
            value = _component_value(record, component)
            if value >= 0:
                start_value = positive_cursor
                end_value = positive_cursor + value
                positive_cursor = end_value
            else:
                start_value = negative_cursor
                end_value = negative_cursor + value
                negative_cursor = end_value

            x_start = scale_x(start_value)
            x_end = scale_x(end_value)
            x = min(x_start, x_end)
            width_rect = max(abs(x_end - x_start), 1.2)
            svg.append(
                f'<rect x="{x:.2f}" y="{bar_y:.2f}" width="{width_rect:.2f}" '
                f'height="{bar_height}" rx="3" fill="{color}" opacity="0.88" stroke="white" '
                f'stroke-width="1"/>'
            )

        total_x = scale_x(record.total_component)
        svg.append(
            f'<line x1="{x_zero:.2f}" y1="{y_center:.2f}" x2="{total_x:.2f}" y2="{y_center:.2f}" '
            f'stroke="#111827" stroke-width="1.2" stroke-dasharray="4 4" opacity="0.55"/>'
        )
        svg.append(
            f'<circle cx="{total_x:.2f}" cy="{y_center:.2f}" r="4.5" fill="#111827"/>'
        )

        svg.append(
            f'<text class="value" text-anchor="middle" x="{value_column_x["electronic"]}" '
            f'y="{y_center + 4:.2f}" fill="{COMPONENT_META[0][2]}">{record.electronic:+.2f}</text>'
        )
        svg.append(
            f'<text class="value" text-anchor="middle" x="{value_column_x["electrostatic"]}" '
            f'y="{y_center + 4:.2f}" fill="{COMPONENT_META[1][2]}">{record.electrostatic:+.2f}</text>'
        )
        svg.append(
            f'<text class="value" text-anchor="middle" x="{value_column_x["lumo"]}" '
            f'y="{y_center + 4:.2f}" fill="{COMPONENT_META[2][2]}">{record.lumo:+.2f}</text>'
        )
        svg.append(
            f'<text class="value" text-anchor="middle" x="{value_column_x["sum"]}" '
            f'y="{y_center + 4:.2f}" fill="#111827">{record.total_component:+.2f}</text>'
        )

    share_top = plot_bottom + 110
    share_left = 260
    share_width = 700
    share_bar_height = 30
    share_gap = 34
    svg.append(f'<text class="section" x="32" y="{share_top - 34}">Selectivity-driving share</text>')
    svg.append(
        f'<text class="subtitle" x="32" y="{share_top - 12}">'
        f'Shares are computed from mean-centered absolute contributions inside each competing set.'
        f'</text>'
    )

    share_sets = (
        ("Initial", initial_metrics),
        ("Follow-up", follow_up_metrics),
        ("Overall", overall_metrics),
    )
    for scope_index, (scope_label, metrics) in enumerate(share_sets):
        y = share_top + scope_index * (share_bar_height + share_gap)
        svg.append(
            f'<text class="label" x="{share_left - 20}" y="{y + 20}" text-anchor="end">'
            f'{scope_label}</text>'
        )
        svg.append(
            f'<rect x="{share_left}" y="{y}" width="{share_width}" height="{share_bar_height}" '
            f'rx="6" fill="none" stroke="#cbd5e1"/>'
        )

        current_x = share_left
        for component, short_label, color in COMPONENT_META:
            segment_width = share_width * metrics[component]["share"]
            svg.append(
                f'<rect x="{current_x:.2f}" y="{y:.2f}" width="{segment_width:.2f}" '
                f'height="{share_bar_height}" fill="{color}" opacity="0.90"/>'
            )
            if segment_width >= 52:
                svg.append(
                    f'<text x="{current_x + segment_width / 2:.2f}" y="{y + 20}" '
                    f'text-anchor="middle" font-size="11" fill="white" font-weight="700">'
                    f'{short_label} {metrics[component]["share"] * 100:.1f}%</text>'
                )
            current_x += segment_width

        svg.append(
            f'<text class="small" x="{share_left + share_width + 24}" y="{y + 20}">'
            f'E {metrics["electronic"]["share"] * 100:.1f}% | '
            f'ES {metrics["electrostatic"]["share"] * 100:.1f}% | '
            f'LUMO {metrics["lumo"]["share"] * 100:.1f}%</text>'
        )

    svg.append("</svg>")
    return "\n".join(svg) + "\n"


def write_group_outputs(group: str, records: list[EntryRecord]) -> dict[str, object]:
    group_dir = OUTPUT_ROOT / group
    group_dir.mkdir(parents=True, exist_ok=True)

    temperature = TEMPERATURE_BY_GROUP[group]
    initial_records = [record for record in records if record.stage == "initial"]
    follow_up_records = [record for record in records if record.stage == "follow_up"]
    overall_metrics = compute_discrimination(records)
    initial_metrics = compute_discrimination(initial_records)
    follow_up_metrics = compute_discrimination(follow_up_records)
    selectivity = simulate_selectivity(records, temperature)
    fastest_initial = min(initial_records, key=lambda record: record.prediction)
    sorted_initial = sorted(initial_records, key=lambda record: record.prediction)
    runner_up_initial = sorted_initial[1]

    max_point_map = {
        item["label"]: {
            "absolute_percent": item["absolute"] * 100,
            "intermediate_pool_percent": item["fraction"] * 100,
        }
        for item in selectivity["intermediates"]
    }
    final_product_map = {
        item["label"]: item["fraction"] * 100
        for item in selectivity["final_products"]
    }
    displayed_intermediate_total = sum(
        round(max_point_map[label]["absolute_percent"], 2)
        for label in INITIAL_LABEL_ORDER
    )

    entry_rows: list[dict[str, object]] = []
    stage_best: dict[str, EntryRecord] = {
        "initial": min(initial_records, key=lambda record: record.prediction),
        "follow_up": min(follow_up_records, key=lambda record: record.prediction),
    }

    for record in records:
        best_in_stage = stage_best[record.stage]
        row = {
            "entry": record.entry,
            "stage": record.stage,
            "name": record.name,
            "prediction_kcal_mol": f"{record.prediction:.6f}",
            "electronic_kcal_mol": f"{record.electronic:.6f}",
            "electrostatic_kcal_mol": f"{record.electrostatic:.6f}",
            "lumo_kcal_mol": f"{record.lumo:.6f}",
            "total_contribution_kcal_mol": f"{record.total_component:.6f}",
            "delta_vs_fastest_stage_kcal_mol": f"{record.prediction - best_in_stage.prediction:.6f}",
            "electronic_delta_vs_fastest_stage_kcal_mol": (
                f"{record.electronic - best_in_stage.electronic:.6f}"
            ),
            "electrostatic_delta_vs_fastest_stage_kcal_mol": (
                f"{record.electrostatic - best_in_stage.electrostatic:.6f}"
            ),
            "lumo_delta_vs_fastest_stage_kcal_mol": f"{record.lumo - best_in_stage.lumo:.6f}",
        }
        entry_rows.append(row)

    csv_row_by_entry: dict[str, list[str]] = {}
    for record in records:
        max_point_percent = (
            max_point_map[record.entry[1:]]["absolute_percent"]
            if record.stage == "initial"
            else 0.0
        )
        final_percent = (
            final_product_map[product_label_from_entry(record.entry)]
            if record.stage == "follow_up"
            else 0.0
        )
        csv_row_by_entry[record.entry] = [
            record.entry,
            record.name,
            format_template_measurement(record.prediction),
            format_template_measurement(record.electronic),
            format_template_measurement(record.electrostatic),
            format_template_measurement(record.lumo),
            format_template_rate(max_point_percent),
            format_template_rate(final_percent),
        ]

    monoalcohol_total_row = [
        "monoalcohol_total",
        "",
        "-",
        "-",
        "-",
        "-",
        format_template_rate(displayed_intermediate_total),
        "0.00",
    ]
    dialcohol_total_row = [
        "dialcohol_total",
        "",
        "-",
        "-",
        "-",
        "-",
        "0.00",
        "100.00",
    ]
    integrated_csv_rows = [
        ["entry", "name", "ΔΔGpredict \n[kcal/mol]", "contribution [kcal/mol]", "", "", "rate [%]", ""],
        ["", "", "", "electronic", "electrostatic", "LUMO", "max point", "final point"],
        *[csv_row_by_entry[f"{group}{suffix}"] for suffix in ("1", "2", "3", "4")],
        monoalcohol_total_row,
        *[csv_row_by_entry[f"{group}{suffix}"] for suffix in ("13", "14", "23", "24", "31", "32", "41", "42")],
        dialcohol_total_row,
    ]

    metric_rows: list[dict[str, object]] = []
    for scope_name, metrics in (
        ("initial", initial_metrics),
        ("follow_up", follow_up_metrics),
        ("overall", overall_metrics),
    ):
        for _, component in COMPONENT_COLUMNS:
            metric_rows.append(
                {
                    "scope": scope_name,
                    "component": component,
                    "selectivity_share_percent": f"{metrics[component]['share'] * 100:.2f}",
                    "discrimination_score_kcal_mol": (
                        f"{metrics[component]['discrimination_score']:.6f}"
                    ),
                    "spread_kcal_mol": f"{metrics[component]['spread']:.6f}",
                    "group_mean_kcal_mol": f"{metrics[component]['center_mean']:.6f}",
                }
            )

    selectivity_rows = [
        {
            "kind": "intermediate_at_tmax",
            "label": item["label"],
            "fraction_percent": f"{item['fraction'] * 100:.2f}",
            "absolute_concentration": f"{item['absolute']:.6f}",
        }
        for item in selectivity["intermediates"]
    ] + [
        {
            "kind": "final_product",
            "label": item["label"],
            "fraction_percent": f"{item['fraction'] * 100:.2f}",
            "absolute_concentration": f"{item['absolute']:.6f}",
        }
        for item in selectivity["final_products"]
    ]

    write_csv(
        group_dir / "entry_contributions.csv",
        fieldnames=list(entry_rows[0].keys()),
        rows=entry_rows,
    )
    write_csv(
        group_dir / "selectivity_contribution_metrics.csv",
        fieldnames=list(metric_rows[0].keys()),
        rows=metric_rows,
    )
    write_csv(
        group_dir / "predicted_selectivity.csv",
        fieldnames=list(selectivity_rows[0].keys()),
        rows=selectivity_rows,
    )
    integrated_filename = f"integrated_summary_{group}.csv"
    write_csv_rows(group_dir / "integrated_summary.csv", integrated_csv_rows)
    write_csv_rows(group_dir / integrated_filename, integrated_csv_rows)
    (group_dir / "contribution_breakdown.svg").write_text(
        build_contribution_svg(
            group=group,
            records=records,
            initial_metrics=initial_metrics,
            follow_up_metrics=follow_up_metrics,
            overall_metrics=overall_metrics,
        ),
        encoding="utf-8",
    )

    table_rows, table_totals = load_integrated_summary_table(group_dir / integrated_filename)
    table_row_by_entry = {
        row["entry"]: row
        for row in table_rows
    }
    table_initial_rows = [
        table_row_by_entry[f"{group}{label}"]
        for label in INITIAL_LABEL_ORDER
    ]
    fastest_table_row = min(table_initial_rows, key=lambda row: row["prediction_value"])
    runner_up_table_row = sorted(
        table_initial_rows, key=lambda row: row["prediction_value"]
    )[1]
    barrier_gap = (
        runner_up_table_row["prediction_value"] - fastest_table_row["prediction_value"]
    )
    component_labels = (
        ("electronic", "electronic_value"),
        ("electrostatic", "electrostatic_value"),
        ("LUMO", "lumo_value"),
    )
    dominant_component_label, dominant_component_key = max(
        component_labels,
        key=lambda item: abs(
            runner_up_table_row[item[1]] - fastest_table_row[item[1]]
        ),
    )
    max_point_text = ", ".join(
        f"{row['entry']} = {row['max_point']}%"
        for row in table_initial_rows
    )
    dominant_intermediate_row = max(
        table_initial_rows,
        key=lambda row: row["max_point_value"],
    )
    second_intermediate_row = sorted(
        table_initial_rows,
        key=lambda row: row["max_point_value"],
        reverse=True,
    )[1]
    final_family_rows = {
        label: table_row_by_entry[f"{group}{suffix}"]
        for label, suffix in (("1-3", "13"), ("1-4", "14"), ("2-3", "23"), ("2-4", "24"))
    }
    final_product_text = ", ".join(
        f"{label} = {final_family_rows[label]['final_point']}%"
        for label in PRODUCT_ORDER
    )
    dominant_product_label = max(
        PRODUCT_ORDER,
        key=lambda label: final_family_rows[label]["final_point_value"],
    )
    runner_up_product_label = sorted(
        PRODUCT_ORDER,
        key=lambda label: final_family_rows[label]["final_point_value"],
        reverse=True,
    )[1]
    dominant_intermediate = selectivity["intermediates"][0]
    dominant_product = selectivity["final_products"][0]

    summary_lines = [
        f"# Test set {group}",
        "",
        f"Substrate family: {base_substrate_name(records[0].name)}.",
        f"All numerical values cited below are taken directly from `{integrated_filename}`.",
        "",
        "## Quantitative Summary",
        "",
        (
            f"Among the initial reductions, the predicted barriers are "
            f"{group}1 = {table_row_by_entry[f'{group}1']['prediction']} kcal/mol, "
            f"{group}2 = {table_row_by_entry[f'{group}2']['prediction']} kcal/mol, "
            f"{group}3 = {table_row_by_entry[f'{group}3']['prediction']} kcal/mol, and "
            f"{group}4 = {table_row_by_entry[f'{group}4']['prediction']} kcal/mol. "
            f"The lowest initial barrier is {fastest_table_row['entry']} "
            f"({fastest_table_row['prediction']} kcal/mol), followed by "
            f"{runner_up_table_row['entry']} ({runner_up_table_row['prediction']} kcal/mol), "
            f"giving a barrier gap of {format_value(barrier_gap)} kcal/mol."
        ),
        "",
        (
            f"The largest component-level separation between these two pathways is found in the "
            f"{dominant_component_label} term, which is {fastest_table_row[dominant_component_key]:.2f} "
            f"kcal/mol for {fastest_table_row['entry']} and "
            f"{runner_up_table_row[dominant_component_key]:.2f} kcal/mol for "
            f"{runner_up_table_row['entry']}. The full contribution set for "
            f"{fastest_table_row['entry']} is electronic {fastest_table_row['electronic']} kcal/mol, "
            f"electrostatic {fastest_table_row['electrostatic']} kcal/mol, and LUMO "
            f"{fastest_table_row['lumo']} kcal/mol."
        ),
        "",
        (
            f"At the max point, the monoalcohol total is {table_totals['monoalcohol_total']['max_point']}%. "
            f"The corresponding distribution is {max_point_text}, so {dominant_intermediate_row['entry']} "
            f"is the dominant monoalcohol intermediate, ahead of {second_intermediate_row['entry']} "
            f"({second_intermediate_row['max_point']}%)."
        ),
        "",
        (
            f"At the final point, the monoalcohol entries are all 0.00% and the dialcohol total is "
            f"{table_totals['dialcohol_total']['final_point']}%. The dialcohol-family distribution is "
            f"{final_product_text}, making the {dominant_product_label} family the major predicted "
            f"outcome, followed by the {runner_up_product_label} family."
        ),
        "",
    ]

    summary_filename = f"summary_{group}.txt"
    (group_dir / summary_filename).write_text("\n".join(summary_lines), encoding="utf-8")
    legacy_summary_path = group_dir / "summary.md"
    if legacy_summary_path.exists():
        legacy_summary_path.unlink()

    return {
        "group": group,
        "substrate_family": base_substrate_name(records[0].name),
        "dominant_intermediate": dominant_intermediate["label"],
        "dominant_intermediate_percent": f"{dominant_intermediate['fraction'] * 100:.2f}",
        "dominant_final_product": dominant_product["label"],
        "dominant_final_product_percent": f"{dominant_product['fraction'] * 100:.2f}",
        "initial_electronic_percent": f"{initial_metrics['electronic']['share'] * 100:.2f}",
        "initial_electrostatic_percent": f"{initial_metrics['electrostatic']['share'] * 100:.2f}",
        "initial_lumo_percent": f"{initial_metrics['lumo']['share'] * 100:.2f}",
        "follow_up_electronic_percent": f"{follow_up_metrics['electronic']['share'] * 100:.2f}",
        "follow_up_electrostatic_percent": f"{follow_up_metrics['electrostatic']['share'] * 100:.2f}",
        "follow_up_lumo_percent": f"{follow_up_metrics['lumo']['share'] * 100:.2f}",
        "overall_electronic_percent": f"{overall_metrics['electronic']['share'] * 100:.2f}",
        "overall_electrostatic_percent": f"{overall_metrics['electrostatic']['share'] * 100:.2f}",
        "overall_lumo_percent": f"{overall_metrics['lumo']['share'] * 100:.2f}",
    }


def main() -> None:
    grouped_records = load_test_records(INPUT_XLSX)
    overview_rows = [
        write_group_outputs(group, grouped_records[group])
        for group in GROUPS
    ]
    write_csv(
        OUTPUT_ROOT / "summary_overview.csv",
        fieldnames=list(overview_rows[0].keys()),
        rows=overview_rows,
    )


if __name__ == "__main__":
    main()
