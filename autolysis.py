
import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import requests
import json

# Print arguments passed to the script
print("Arguments passed:", sys.argv)
print("Starting analysis script")

# Verify Environment Variable
if "AIPROXY_TOKEN" not in os.environ:
    print("Error: AIPROXY_TOKEN environment variable is not set.")
    sys.exit(1)

AIPROXY_TOKEN = os.environ["AIPROXY_TOKEN"]

# Function to load dataset
def load_dataset(file_path):
    try:
        return pd.read_csv(file_path,encoding='ISO-8859-1')
    except Exception as e:
        print(f"Error loading dataset: {e}")
        sys.exit(1)

# Function to analyze the dataset
def analyze_dataset(df):
    summary = {
        "columns": list(df.columns),
        "data_types": df.dtypes.to_dict(),
        "missing_values": df.isnull().sum().to_dict(),
        "summary_stats": df.describe(include="all").to_dict()
    }
    return summary

# Function to visualize data
def visualize_data(df, output_folder):
    # Select only numeric columns for correlation
    numeric_df = df.select_dtypes(include=['number'])

    # Example: Correlation heatmap
    if not numeric_df.empty:
        plt.figure(figsize=(10, 8))
        sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm")
        plt.title("Correlation Heatmap")
        heatmap_path = os.path.join(output_folder, "heatmap.png")
        plt.savefig(heatmap_path)
        plt.close()
        return [heatmap_path]
    else:
        print("No numeric data available for correlation analysis.")
        return []

# Function to send request to AI Proxy (GPT-4o-Mini)
def generate_narration(prompt):
    url = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [{"role": "user", "content": prompt}],
    }
    try:
        response = requests.post(url, headers=headers, data=json.dumps(data))
        response.raise_for_status()
        return response.json()['choices'][0]['message']['content']
    except Exception as e:
        print(f"Error generating narration: {e}")
        sys.exit(1)

# Main function
def main():
    if len(sys.argv) != 2:
        print("Usage: uv run autolysis.py dataset.csv")
        sys.exit(1)

    file_path = sys.argv[1]
    output_folder = os.path.splitext(file_path)[0]
    os.makedirs(output_folder, exist_ok=True)

    # Load dataset
    df = load_dataset(file_path)

    # Analyze dataset
    summary = analyze_dataset(df)

    # Visualize data
    chart_paths = visualize_data(df, output_folder)

    # Generate narration
    prompt = f"""
    I analyzed the dataset '{file_path}' with the following characteristics:
    {summary}.
    Based on the analysis, here is the narrative:
    """
    story = generate_narration(prompt)

    # Write to README.md
    readme_path = os.path.join(output_folder, "README.md")
    with open(readme_path, "w") as f:
        f.write(f"# Automated Analysis Report\n\n## Summary\n{summary}\n\n")
        f.write(f"## Story\n{story}\n\n")
        f.write("## Visualizations\n")
        for chart in chart_paths:
            f.write(f"![Visualization]({chart})\n")

    print(f"Analysis complete. Results saved in {output_folder}")

if __name__ == "__main__":
    main()

