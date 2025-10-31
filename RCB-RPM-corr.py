import pandas as pd
import numpy as np
import glob
import os
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog

def process_rpm(rpm_series):
    """Processes RPM data step by step, returns each step as a separate Series."""
    
    rpm_original = rpm_series.reset_index(drop=True)

    found_start = False
    start_index = None
    for window_size in [5, 4, 3, 2]:
        for i in range(len(rpm_original) - window_size + 1):
            window = rpm_original.iloc[i:i+window_size]
            if float(window.max()) - float(window.min()) <= 10:
                start_index = i
                found_start = True
                break
        if found_start:
            break

    if not found_start:
        return {
            'rpm_original': rpm_original,
            'rpm_after_minigroup': rpm_original,
            'rpm_cleaned': rpm_original,
            'rpm_mean': rpm_original.mean()
        }

    rpm_after_minigroup = rpm_original.iloc[start_index:].reset_index(drop=True)

    rpm_cleaned = [rpm_after_minigroup.iloc[0]]
    for val in rpm_after_minigroup.iloc[1:]:
        if abs(val - rpm_cleaned[-1]) <= 10:
            rpm_cleaned.append(val)
    rpm_cleaned = pd.Series(rpm_cleaned)

    rpm_mean = rpm_cleaned.mean()

    return {
        'rpm_original': rpm_original,
        'rpm_after_minigroup': rpm_after_minigroup,
        'rpm_cleaned': rpm_cleaned,
        'rpm_mean': rpm_mean
    }

def process_file(filepath, output_plot_dir, output_data_dir):
    df = pd.read_csv(filepath)
    df = df.iloc[15:].reset_index(drop=True)

    esc_col = 'ESC B (µs)'
    rpm_col = 'Motor Optical Speed B (RPM)'
    torque_col = 'Torque B (N·m)'
    el_power_col = 'Electrical Power B (W)'
    thrust_col = 'Thrust B (N)'
    mech_power_col = 'Mechanical Power B (W)'
    motor_eff_col = 'Motor Efficiency B (%)'
    prop_eff_col = 'Propeller Mech. Efficiency B (N/W)'

    grouped = df.groupby(esc_col)

    for esc_value, group in grouped:
        if len(group) < 5:
            continue

        rpm_series = group[rpm_col].dropna().reset_index(drop=True)
        if rpm_series.empty:
            continue

        processed_rpm = process_rpm(rpm_series)

        # Plotting
        plt.figure(figsize=(10,6))
        plt.plot(processed_rpm['rpm_original'], label='Original', alpha=0.7)
        plt.plot(processed_rpm['rpm_after_minigroup'], label='After Minigroup', alpha=0.7)
        plt.plot(processed_rpm['rpm_cleaned'], label='Cleaned', alpha=0.7)
        plt.axhline(processed_rpm['rpm_mean'], color='black', linestyle='--', label=f"Mean: {processed_rpm['rpm_mean']:.2f}")
        plt.title(f'File: {os.path.basename(filepath)} | ESC B: {esc_value}')
        plt.xlabel('Row Index')
        plt.ylabel('RPM B')
        plt.legend()
        plt.grid(True)
        plot_filename = os.path.join(output_plot_dir, f"{os.path.basename(filepath).replace('.csv','')}_ESC_{esc_value}_rpm_plot.png")
        plt.savefig(plot_filename)
        plt.close()

        # Replace RPM in the dataframe with the new mean RPM for that group
        group_index = group.index
        df.loc[group_index, rpm_col] = processed_rpm['rpm_mean']

    # Now recalculate derived columns for the entire dataframe
    # Mechanical Power B (W) = (2 * pi * rpm * torque) / 60
    df[mech_power_col] = (2 * np.pi * df[rpm_col] * df[torque_col]) / 60

    # Motor Efficiency B (%) = (Mechanical Power B) / (Electrical Power B) * 100
    df[motor_eff_col] = np.where(df[el_power_col] != 0, (df[mech_power_col] / df[el_power_col]) * 100, np.nan)

    # Propeller Mech. Efficiency B (N/W) = (Thrust B) / (Mechanical Power B)
    df[prop_eff_col] = np.where(df[mech_power_col] != 0, df[thrust_col] / df[mech_power_col], np.nan)

    # Save the corrected dataframe to the output folder
    output_filename = os.path.basename(filepath).replace('.csv', '.csv')
    output_path = os.path.join(output_data_dir, output_filename)
    df.to_csv(output_path, index=False)

def main():
    # Select raw directory using file dialog
    root = tk.Tk()  
    root.withdraw()
    # Prompt user to select the 'raw' directory
    raw_dir = filedialog.askdirectory(title="Select the 'raw' directory containing CSV files")
    if not raw_dir:
        print("No directory selected. Exiting.")
        return
    csv_files = glob.glob(os.path.join(raw_dir, '*.csv'))
    # Create output directories in the folder which was chosen by the user
    output_plot_dir = os.path.join(raw_dir, 'plots')
    output_data_dir = os.path.join(raw_dir, 'corrected_csvs')
    os.makedirs(output_plot_dir, exist_ok=True)
    os.makedirs(output_data_dir, exist_ok=True)

    for file in csv_files:
        process_file(file, output_plot_dir, output_data_dir)

    print(f"Processing completed. Plots saved in '{output_plot_dir}' and corrected CSVs in '{output_data_dir}'.")

if __name__ == '__main__':
    main()
