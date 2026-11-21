"""Data Cleaning AI Agent - Profiler -> Cleaner -> Reporter (runs offline)."""
import sys
import numpy as np
import pandas as pd


class ProfilerAgent:
    """Inspects the dataset and measures its quality."""
    def __init__(self, df):
        self.df = df

    def outlier_mask(self, col):
        s = self.df[col]
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        return (s < q1 - 1.5 * iqr) | (s > q3 + 1.5 * iqr)

    def profile(self):
        df = self.df
        num_cols = df.select_dtypes("number").columns
        missing = int(df.isna().sum().sum())
        dups = int(df.duplicated().sum())
        outliers = {c: int(self.outlier_mask(c).sum()) for c in num_cols}
        cells = df.shape[0] * df.shape[1]
        penalty = missing / cells * 40 + dups / len(df) * 30 + sum(outliers.values()) / len(df) * 30
        return {"rows": len(df), "cols": df.shape[1], "missing": missing,
                "missing_by_col": df.isna().sum()[lambda x: x > 0].to_dict(),
                "duplicates": dups, "outliers": outliers,
                "score": round(max(0, 100 - penalty), 1)}


class CleanerAgent:
    """Applies cleaning steps in order and records every action."""
    GENDER_MAP = {"m": "Male", "male": "Male", "f": "Female", "female": "Female"}

    DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%d-%b-%Y", "%d-%m-%Y"]
    RULES = {"age": (15, 60), "score": (0, 100)}  # valid ranges (domain knowledge)

    @classmethod
    def parse_date(cls, v):
        for f in cls.DATE_FORMATS:
            try:
                return pd.to_datetime(v, format=f).strftime("%Y-%m-%d")
            except (ValueError, TypeError):
                continue
        return None

    def __init__(self, df, outlier_strategy="cap"):
        self.df, self.strategy, self.log = df.copy(), outlier_strategy, []

    def run(self):
        df = self.df
        n = len(df)
        df = df.drop_duplicates().reset_index(drop=True)
        self.log.append(f"Removed {n - len(df)} duplicate row(s).")

        for c in df.select_dtypes(include=["object", "string"]).columns:
            df[c] = df[c].str.strip().str.title() if c != "branch" else df[c].str.strip().str.upper()
        if "gender" in df:
            df["gender"] = df["gender"].str.lower().map(self.GENDER_MAP).fillna(df["gender"])
        self.log.append("Trimmed whitespace and normalised text casing (names, gender, branch).")

        for c in df.columns:
            if "date" in c:
                df[c] = df[c].apply(self.parse_date)
                self.log.append(f"Standardised '{c}' to YYYY-MM-DD.")

        # impossible values (e.g. age 150, score 350) -> treated as missing
        for c, (lo, hi) in self.RULES.items():
            if c in df:
                bad = (df[c] < lo) | (df[c] > hi)
                if bad.any():
                    df.loc[bad, c] = np.nan
                    self.log.append(f"Marked {int(bad.sum())} impossible value(s) in '{c}' (valid {lo}-{hi}) as missing.")

        # statistical outliers (IQR rule) on the remaining values
        for c in df.select_dtypes("number").columns:
            if c == "student_id":
                continue
            q1, q3 = df[c].quantile(.25), df[c].quantile(.75)
            lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
            bad = (df[c] < lo) | (df[c] > hi)
            if bad.any():
                if self.strategy == "cap":
                    df[c] = df[c].clip(lo, hi)
                elif self.strategy == "remove":
                    df = df[~bad | df[c].isna()]
                self.log.append(f"Outliers in '{c}': {int(bad.sum())} found, strategy='{self.strategy}' "
                                f"(bounds {lo:.1f} to {hi:.1f}).")

        for c in df.columns:
            k = int(df[c].isna().sum())
            if k:
                if pd.api.types.is_numeric_dtype(df[c]):
                    df[c] = df[c].fillna(df[c].median()); how = "median"
                else:
                    df[c] = df[c].fillna(df[c].mode()[0]); how = "mode"
                self.log.append(f"Filled {k} missing value(s) in '{c}' using {how}.")
        if "age" in df:
            df["age"] = df["age"].round().astype(int)
        return df.reset_index(drop=True), self.log


class ReporterAgent:
    """Turns the before/after numbers into a plain-English report."""
    def __init__(self, before, after, log):
        self.b, self.a, self.log = before, after, log

    def write(self):
        lines = [f"Dataset: {self.b['rows']} rows x {self.b['cols']} columns.",
                 f"Quality score: {self.b['score']} -> {self.a['score']} / 100.",
                 f"Missing cells: {self.b['missing']} -> {self.a['missing']}.",
                 f"Duplicate rows: {self.b['duplicates']} -> {self.a['duplicates']}.",
                 "", "Actions taken:"] + [f"  - {x}" for x in self.log]
        return "\n".join(lines)


if __name__ == "__main__":
    src = sys.argv[1] if len(sys.argv) > 1 else "messy_students.csv"
    raw = pd.read_csv(src)
    before = ProfilerAgent(raw).profile()
    clean, log = CleanerAgent(raw).run()
    after = ProfilerAgent(clean).profile()
    clean.to_csv("cleaned_output.csv", index=False)
    report = ReporterAgent(before, after, log).write()
    open("report.txt", "w").write(report)
    print(report)
