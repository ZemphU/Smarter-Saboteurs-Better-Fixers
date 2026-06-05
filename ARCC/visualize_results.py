import re
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def parse_results(filepath, condition_name):
    """
    Parses a result text file and returns a list of dictionaries.
    """
    results = []
    current_model = None
    
    with open(filepath, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith("Model:"):
                current_model = line.split("Model:")[1].strip()
            elif line.startswith("Scenario:"):
                parts = line.split("|")
                scenario = parts[0].split("Scenario:")[1].strip()
                # pass_info = parts[1].strip() # Passed: X/Y
                rate_str = parts[2].split("Rate:")[1].strip().replace("%", "")
                rate = float(rate_str)
                
                # Extract size for sorting
                # Assuming format google/gemma-3-270m-it
                size_match = re.search(r'-(\d+b|\d+m)-', current_model)
                size_val = 0
                size_label = "Unknown"
                if size_match:
                    size_label = size_match.group(1)
                    if 'b' in size_label:
                        size_val = float(size_label.replace('b', '')) * 1000
                    elif 'm' in size_label:
                        size_val = float(size_label.replace('m', ''))
                
                results.append({
                    "Model": current_model,
                    "Size_Label": size_label,
                    "Size_Val": size_val,
                    "Scenario": scenario,
                    "Condition": condition_name,
                    "Pass_Rate": rate
                })
    return results

def main():
    # File paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    nofix_path = os.path.join(base_dir, "Results", "gemma-native-results-nofix.txt")
    qafix_path = os.path.join(base_dir, "Results", "gemma-native-results-qafix.txt")
    output_dir = os.path.join(base_dir, "Results", "plots")
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Parse data
    data_nofix = parse_results(nofix_path, "No Fix")
    data_qafix = parse_results(qafix_path, "QA Fix")
    
    all_data = data_nofix + data_qafix
    df = pd.DataFrame(all_data)
    
    # Sort by size
    df = df.sort_values(by="Size_Val")
    
    # --- Visualization 1: Grouped Bar Chart (All Conditions) ---
    plt.figure(figsize=(14, 8))
    sns.set_theme(style="whitegrid")
    
    # Create a unified 'Hue' column for better grouping
    df['Group'] = df['Condition'] + " - " + df['Scenario']
    
    # We want a specific order: No Fix-Control, No Fix-Malicious, QA Fix-Control, QA Fix-Malicious
    hue_order = ["No Fix - Control", "No Fix - Malicious", "QA Fix - Control", "QA Fix - Malicious"]
    
    # Define a custom color palette for better contrast between conditions
    custom_palette = {
        "No Fix - Control": "#1f77b4",       # Blue
        "No Fix - Malicious": "#d62728",     # Red
        "QA Fix - Control": "#2ca02c",       # Green
        "QA Fix - Malicious": "#ff7f0e"      # Orange
    }
    
    # Using 'Model' as x-axis might be cluttered if names are long, maybe just use Size Label if models are same family
    # Let's clean up model names for the X axis
    df['Model_Short'] = df['Model'].apply(lambda x: x.split('/')[-1])
    
    g = sns.barplot(
        data=df, 
        x="Model_Short", 
        y="Pass_Rate", 
        hue="Group", 
        hue_order=hue_order,
        palette=custom_palette
    )
    
    plt.title("Gemma Model Performance: Control vs Malicious (With and Without QA Fix)", fontsize=16)
    plt.ylabel("Pass Rate (%)", fontsize=12)
    plt.xlabel("Model", fontsize=12)
    plt.xticks(rotation=45)
    plt.legend(title="Condition - Scenario", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gemma_native_performance_comparison_bar.png"))
    print(f"Saved gemma_native_performance_comparison_bar.png to {output_dir}")
    plt.close()

    # --- Visualization 2: Scaling Line Plot ---
    plt.figure(figsize=(12, 6))
    
    # Plot lines for each group
    sns.lineplot(
        data=df, 
        x="Model_Short", 
        y="Pass_Rate", 
        hue="Group", 
        hue_order=hue_order,
        style="Group",
        markers=True, 
        dashes=False,
        palette=custom_palette,
        linewidth=2.5
    )
    
    plt.title("Performance Scaling by Model Size", fontsize=16)
    plt.ylabel("Pass Rate (%)", fontsize=12)
    plt.xlabel("Model Size", fontsize=12)
    plt.grid(True, linestyle="--", alpha=0.7)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "gemma_native_performance_scaling_line.png"))
    print(f"Saved gemma_native_performance_scaling_line.png to {output_dir}")
    plt.close()

    # --- Visualization 3: Malicious Impact (Drop in Performance) ---
    # Calculate the drop: Control - Malicious for each Model and Condition
    
    impact_data = []
    models = df['Model_Short'].unique()
    conditions = ['No Fix', 'QA Fix']
    
    for model in models:
        for cond in conditions:
            subset = df[(df['Model_Short'] == model) & (df['Condition'] == cond)]
            if len(subset) == 2:
                control = subset[subset['Scenario'] == 'Control']['Pass_Rate'].values[0]
                malicious = subset[subset['Scenario'] == 'Malicious']['Pass_Rate'].values[0]
                drop = control - malicious
                impact_data.append({
                    'Model': model,
                    'Condition': cond,
                    'Performance Drop': drop # Positive means Malicious hurt performance
                })
    
    if impact_data:
        df_impact = pd.DataFrame(impact_data)
        
        plt.figure(figsize=(10, 6))
        sns.barplot(
            data=df_impact,
            x="Model",
            y="Performance Drop",
            hue="Condition",
            palette="rocket"
        )
        
        plt.title("Impact of Malicious Agent (Control Rate - Malicious Rate)", fontsize=16)
        plt.ylabel("Performance Drop (% points)", fontsize=12)
        plt.xlabel("Model", fontsize=12)
        plt.axhline(0, color='black', linewidth=0.8)
        plt.legend(title="Condition")
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, "gemma_native_malicious_impact_drop.png"))
        print(f"Saved gemma_native_malicious_impact_drop.png to {output_dir}")
        plt.close()

if __name__ == "__main__":
    main()
