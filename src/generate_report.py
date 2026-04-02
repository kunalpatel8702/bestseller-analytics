try:
    from ydata_profiling import ProfileReport
except ImportError:
    ProfileReport = None
import sys

def generate_report(data_path="data/cleaned.csv", output_path="reports/profile_report.html"):
    if ProfileReport is None:
        print("[ERROR] 'ydata-profiling' not found. Please install with: pip install ydata-profiling")
        return

    print(f"[INFO] Generating Profiling Report (ydata-profiling)...")
    try:
        import pandas as pd
        df = pd.read_csv(data_path)
        profile = ProfileReport(df, title="Book Bestsellers Profiling Report", explorative=True)
        profile.to_file(output_path)
        print(f"[SUCCESS] Report saved to {output_path}")
    except Exception as e:
        print(f"[ERROR] Profiling failed: {str(e)}")

if __name__ == "__main__":
    generate_report()
