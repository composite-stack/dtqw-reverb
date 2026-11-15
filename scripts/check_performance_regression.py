#!/usr/bin/env python3
import argparse
import json
import os
import sys

def format_row(cols, widths, align="<"):
    out = []
    for c, w in zip(cols, widths):
        s = str(c)
        if len(s) > w:
            s = s[:w-1] + "…"
        out.append(f"{s:{align}{w}}")
    return "  ".join(out)

def main():
    parser = argparse.ArgumentParser(description="DTQW DSP Performance Regression Check")
    parser.add_argument("--current", required=True, help="Path to current benchmark JSON")
    parser.add_argument("--baseline", required=True, help="Path to baseline benchmark JSON")
    parser.add_argument("--tolerance", type=float, default=25.0, help="Max allowed % slowdown (default: 25.0)")
    parser.add_argument("--metric", choices=["median_ms_per_sec", "min_ms_per_sec"], default="median_ms_per_sec",
                        help="Metric to compare (default: median_ms_per_sec)")
    parser.add_argument("--summary", help="Optional path to write GitHub Actions step summary Markdown")
    args = parser.parse_args()

    if not os.path.exists(args.current):
        print(f"Error: Current benchmark file not found: {args.current}", file=sys.stderr)
        sys.exit(2)

    if not os.path.exists(args.baseline):
        print(f"Error: Baseline benchmark file not found: {args.baseline}", file=sys.stderr)
        sys.exit(2)

    with open(args.current) as f:
        current_data = json.load(f).get("scenarios", {})

    with open(args.baseline) as f:
        baseline_data = json.load(f).get("scenarios", {})

    headers = ["Szenario", "Baseline (ms/s)", "Aktuell (ms/s)", "Delta %", "Baseline RTF", "Aktuell RTF", "Status"]
    widths  = [26,         17,               16,             10,        14,             13,            14]

    print("=" * 115)
    print("  DTQW Reverb - DSP Performance Regression Check (Toleranz: ±{:.1f}%)".format(args.tolerance))
    print("=" * 115)
    print(format_row(headers, widths))
    print("-" * 115)

    regressions = 0
    markdown_rows = []

    for scenario_name, base_info in baseline_data.items():
        if scenario_name not in current_data:
            print(f"Warning: Scenario {scenario_name} missing from current run!", file=sys.stderr)
            status_str = "MISSING"
            regressions += 1
            continue

        curr_info = current_data[scenario_name]
        base_val = base_info.get(args.metric, 0.0)
        curr_val = curr_info.get(args.metric, 0.0)

        base_rtf = base_info.get("realtime_factor", 0.0)
        curr_rtf = curr_info.get("realtime_factor", 0.0)

        if base_val <= 0.0:
            diff_pct = 0.0
        else:
            diff_pct = ((curr_val - base_val) / base_val) * 100.0

        if diff_pct > args.tolerance:
            status_str = "❌ REGRESSION"
            regressions += 1
        elif diff_pct < -args.tolerance:
            status_str = "🚀 FASTER"
        else:
            status_str = "✅ PASS"

        delta_sign = "+" if diff_pct > 0 else ""
        delta_str = f"{delta_sign}{diff_pct:.1f}%"

        row = [
            scenario_name,
            f"{base_val:.2f}",
            f"{curr_val:.2f}",
            delta_str,
            f"{base_rtf:.1f}x",
            f"{curr_rtf:.1f}x",
            status_str
        ]
        print(format_row(row, widths))

        markdown_rows.append(
            f"| `{scenario_name}` | {base_val:.2f} ms | {curr_val:.2f} ms | **{delta_str}** | {base_rtf:.1f}x | {curr_rtf:.1f}x | {status_str} |"
        )

    print("-" * 115)

    if regressions > 0:
        print(f"\n❌ FEHLER: {regressions} Szenario(n) weisen eine deutliche Performance-Regression auf (> {args.tolerance:.1f}% Verlangsamung)!\n")
    else:
        print(f"\n✅ ERFOLG: Alle Szenarien liegen innerhalb der Performance-Toleranz (≤ {args.tolerance:.1f}% Verlangsamung).\n")

    if args.summary:
        with open(args.summary, "w") as sf:
            sf.write("### ⏱️ DTQW DSP Performance Regression Check\n\n")
            if regressions > 0:
                sf.write(f"❌ **FAILED**: {regressions} Szenario(n) überschreiten das Regressions-Limit von +{args.tolerance:.1f}%!\n\n")
            else:
                sf.write(f"✅ **PASSED**: Alle DSP-Szenarien innerhalb der Toleranz (Toleranzlimit: +{args.tolerance:.1f}%).\n\n")

            sf.write("| Szenario | Baseline (ms/s) | Aktuell (ms/s) | Delta % | Baseline RTF | Aktuell RTF | Status |\n")
            sf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: |\n")
            for mr in markdown_rows:
                sf.write(mr + "\n")
            sf.write("\n")

    sys.exit(1 if regressions > 0 else 0)

if __name__ == "__main__":
    main()
