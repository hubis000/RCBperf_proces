import pandas as pd
import numpy as np
import os
import datetime
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, ttk
from collections import defaultdict
from matplotlib import colors as mcolors
import plotly.graph_objects as go
from tkinter import Tk, filedialog
from antfunctions import antfunctions
import json

# File to store previous inputs
SAVE_FILE = "last_inputs.json"

# Column alias mapping (primary first, fallback(s) next)
COLUMN_ALIASES = {
    "ESC B (µs)": ["ESC B (µs)", "Powertrain 1 - ESC throttle (μs)"],
    "Time (s)": ["Time (s)"],
    "Thrust B (N)": ["Thrust B (N)", "Powertrain 1 - force Fz (thrust) (N)"],
    "Torque B (N·m)": ["Torque B (N·m)", "Powertrain 1 - torque MZ (torque) (N⋅m)"],
    "Mechanical Power B (W)": ["Mechanical Power B (W)", "Powertrain 1 - mechanical power (W)"],
    "Motor Optical Speed B (RPM)": ["Motor Optical Speed B (RPM)", "Powertrain 1 - rotation speed (rpm)"],
}

def get_column(df, logical_name):
    """Return Series for the first available alias of a logical column name."""
    for alias in COLUMN_ALIASES.get(logical_name, []):
        if alias in df.columns:
            return df[alias]
    raise KeyError(f"None of the aliases for '{logical_name}' found. Tried {COLUMN_ALIASES.get(logical_name, [])}")


def load_last_inputs():
    if os.path.exists(SAVE_FILE):
        with open(SAVE_FILE, "r") as f:
            return json.load(f)
    return {}

def save_inputs():
    data = {
        "fit_curve": fit_curve_var.get(),
        "export_table_data": export_table_data_var.get(),
        "OP": OP_entry.get(),
        "diameter": diameter_entry.get(),
        "max_rpm": max_rpm_entry.get(),
        "density": density_entry.get(),
        "cut_off_from_start": cut_off_from_start_entry.get(),
        "cut_off_from_end": cut_off_from_end_entry.get(),
        "temperature": temperature_entry.get(),
        "export_extrapolated_data": export_extrapolated_data_var.get()
    }
    with open(SAVE_FILE, "w") as f:
        json.dump(data, f)

def submit():
    global fit_curve, export_table_data, OP, diameter, max_rpm, density, cut_off_from_end, cut_off_from_start, temperature, export_extrapolated_data
    fit_curve = fit_curve_var.get()
    export_table_data = export_table_data_var.get()
    OP = int(OP_entry.get())
    diameter = diameter_entry.get()
    max_rpm = max_rpm_entry.get()
    density = float(density_entry.get())
    cut_off_from_start = int(cut_off_from_start_entry.get())
    cut_off_from_end = int(cut_off_from_end_entry.get())
    export_extrapolated_data = export_extrapolated_data_var.get()
    if temperature_entry.get() == "":
        temperature = ""
    else:
        temperature = float(temperature_entry.get())
    
    save_inputs()  # Save inputs before quitting

    # Print the values to verify
    print(f"fit_curve: {fit_curve}")
    print(f"export_table_data: {export_table_data}")
    print(f"OP: {OP}")
    print(f"diameter: {diameter}")
    print(f"max_rpm: {max_rpm}")
    print(f"density: {density}")
    print(f"cut_off_from_start: {cut_off_from_start}")
    print(f"cut_off_from_end: {cut_off_from_end}")
    print(f"temperature: {temperature}")
    print(f"export_extrapolated_data: {export_extrapolated_data}")
    
    root.quit()

# Helper functions for alias-based column lookup
def find_column(df, logical_name):
    """
    Return the actual column name present in df for the given logical_name,
    trying aliases in order. Raises KeyError if none found.
    """
    if logical_name not in COLUMN_ALIASES:
        raise KeyError(f"No alias mapping defined for logical name '{logical_name}'")
    for alias in COLUMN_ALIASES[logical_name]:
        if alias in df.columns:
            return alias
    raise KeyError(f"None of the aliases for '{logical_name}' found in DataFrame. Tried: {COLUMN_ALIASES[logical_name]}")

def get_series(df, logical_name):
    """Return the Series from df corresponding to logical_name (via alias lookup)."""
    col = find_column(df, logical_name)
    return df[col]

# ----------------------- GUI form -----------------------
# Load previous inputs
last_inputs = load_last_inputs()

# Create the main window
root = tk.Tk()
root.title("Input Form")

# Create variables for the form values
fit_curve_var = tk.StringVar(value=last_inputs.get("fit_curve", "Yes"))
export_table_data_var = tk.StringVar(value=last_inputs.get("export_table_data", "Yes"))
export_extrapolated_data_var = tk.StringVar(value=last_inputs.get("export_extrapolated_data", "Yes"))

def create_entry(label, row, key, default):
    ttk.Label(root, text=label).grid(row=row, column=0, padx=10, pady=5)
    entry = ttk.Entry(root)
    entry.insert(0, last_inputs.get(key, default))
    entry.grid(row=row, column=1, padx=10, pady=5)
    return entry

# Create and place the form fields
ttk.Label(root, text="Fit Curve:").grid(row=0, column=0, padx=10, pady=5)
fit_curve_menu = ttk.Combobox(root, textvariable=fit_curve_var, values=["Yes", "No"])
fit_curve_menu.grid(row=0, column=1, padx=10, pady=5)
ttk.Label(root, text="Plot a fitted curve in the exported plots.").grid(row=0, column=2, padx=10, pady=5)

ttk.Label(root, text="Export Table Data:").grid(row=1, column=0, padx=10, pady=5)
export_table_data_menu = ttk.Combobox(root, textvariable=export_table_data_var, values=["Yes", "No"])
export_table_data_menu.grid(row=1, column=1, padx=10, pady=5)
ttk.Label(root, text="Export a table with the averaged data for the report.").grid(row=1, column=2, padx=10, pady=5)

OP_entry = create_entry("OP:", 2, "OP", "0")
ttk.Label(root, text="Work in Progress. (Will plot the Operational point on the plots)").grid(row=2, column=2, padx=10, pady=5)

diameter_entry = create_entry("Diameter [inches]:", 3, "diameter", "28")
ttk.Label(root, text="Diameter of the measured propeller. If left empty, will take value from file name").grid(row=3, column=2, padx=10, pady=5)

max_rpm_entry = create_entry("Max RPM:", 4, "max_rpm", "")
ttk.Label(root, text="Maximum RPM on x axis on the plots. If left empty, will take 0.7M tip speed.").grid(row=4, column=2, padx=10, pady=5)

density_entry = create_entry("Air Density:", 5, "density", "")
ttk.Label(root, text="Air density during measurement.").grid(row=5, column=2, padx=10, pady=5)

cut_off_from_start_entry = create_entry("Cut Off From Start [s]:", 6, "cut_off_from_start", "3")
ttk.Label(root, text="Cut off during step averaging").grid(row=6, column=2, padx=10, pady=5)

cut_off_from_end_entry = create_entry("Cut Off From End [s]:", 7, "cut_off_from_end", "1")
ttk.Label(root, text="Cut off during step averaging").grid(row=7, column=2, padx=10, pady=5)

temperature_entry = create_entry("Temperature [°C]:", 8, "temperature", "")
ttk.Label(root, text="OPTIONAL - Temperature during measurement.").grid(row=8, column=2, padx=10, pady=5)

ttk.Label(root, text="Export Extrapolated Data:").grid(row=9, column=0, padx=10, pady=5)
export_extrapolated_data_menu = ttk.Combobox(root, textvariable=export_extrapolated_data_var, values=["Yes", "No"])
export_extrapolated_data_menu.grid(row=9, column=1, padx=10, pady=5)
ttk.Label(root, text="Export a csv file with the extrapolated data.").grid(row=9, column=2, padx=10, pady=5)

# Create and place the submit button
submit_button = ttk.Button(root, text="Submit", command=submit)
submit_button.grid(row=10, columnspan=3, pady=(10, 20))

# Run the application
root.mainloop()

# ------------------ Prepare folders and inputs ------------------
# Open directory dialog to select the input folder
main = filedialog.askdirectory(title="Select folder with measured data")
# get name of the folder
main_name = os.path.basename(main)
print("\n")
print(f"Selected folder: {main}")

FS_step_folder = os.path.join(os.path.dirname(main), "02-FS_step")
FS_avg_folder = os.path.join(os.path.dirname(main), "03-FS_avg")
FS_data = os.path.join(os.path.dirname(main), "04-csv_export")
if not os.path.exists(FS_data):
    os.makedirs(FS_data)
if not os.path.exists(FS_step_folder):
    os.makedirs(FS_step_folder)
if not os.path.exists(FS_avg_folder):
    os.makedirs(FS_avg_folder)
plots_folder = os.path.join(os.path.dirname(main), "Plots")
if not os.path.exists(plots_folder):
    os.makedirs(plots_folder)
sealevel_folder = os.path.join(os.path.dirname(main), "03-FS_avg-0mISA")
if not os.path.exists(sealevel_folder):
    os.makedirs(sealevel_folder)   

density = float(density)

# ------------------ Step average creation ================================
for file in os.listdir(main):
    if file.endswith(".csv"):
        file_path = os.path.join(main, file)
        df = pd.read_csv(file_path)

        # Drop rows where the value of the ESC throttle column is empty
        esc_col = find_column(df, "ESC B (µs)")
        df = df.dropna(subset=[esc_col])

        # Find the last occurrence of the highest value in the ESC column
        max_value = df[esc_col].max()
        last_max_index = df[df[esc_col] == max_value].index[-1]

        # Keep only the rows up to the last occurrence of the highest value
        df = df.loc[:last_max_index]

        # Group by ESC and apply the cut-off within each group
        time_col = find_column(df, "Time (s)")
        def apply_cutoff(group):
            group = group.copy()
            # Reset time to start from 0
            group[time_col] = group[time_col] - group[time_col].min()
            max_time = group[time_col].max()
            return group[(group[time_col] >= cut_off_from_start) & (group[time_col] <= max_time - cut_off_from_end)]

        df = df.groupby(esc_col).apply(apply_cutoff).reset_index(drop=True)

        print(df)

        # Group by ESC and calculate the mean for each group
        avg_df = df.groupby(esc_col).mean().reset_index()

        print(avg_df)

        # Save the modified DataFrame back to a CSV file
        avg_df.to_csv(os.path.join(FS_step_folder, file), index=False)

# ------------------ Combine averages =====================================
# Dictionary to hold dataframes grouped by file name prefix
file_groups = defaultdict(list)

# Group files by name prefix (excluding last two digits)
for file in os.listdir(FS_step_folder):
    if file.endswith(".csv"):
        prefix = file[:-6]  # Exclude last two digits and file extension
        file_groups[prefix].append(file)

# Process each group of files
for prefix, files in file_groups.items():
    combined_df = pd.DataFrame()

    for file in files:
        file_path = os.path.join(FS_step_folder, file)
        df = pd.read_csv(file_path)
        if combined_df.empty:
            combined_df = df
        else:
            combined_df = pd.concat([combined_df, df], ignore_index=True)

    # Group by ESC and calculate the mean for each group
    esc_col_name = find_column(combined_df, "ESC B (µs)")
    avg_df = combined_df.groupby(esc_col_name).mean().reset_index()

    # Save the combined average DataFrame to a new CSV file
    avg_file_path = os.path.join(FS_avg_folder, f"{prefix}avg.csv")
    avg_df.to_csv(avg_file_path, index=False)

print("Averaging complete.")

# ------------------ Create average data at 0m ISA --------------------------
density_ratio = 1.225 / density
for file in os.listdir(FS_avg_folder):
    if file.endswith(".csv"):
        file_path = os.path.join(FS_avg_folder, file)
        df = pd.read_csv(file_path)

        thrust_col = find_column(df, "Thrust B (N)")
        torque_col = find_column(df, "Torque B (N·m)")
        power_col = find_column(df, "Mechanical Power B (W)")
        rpm_col = find_column(df, "Motor Optical Speed B (RPM)")

        # Multiply the specified columns by the density ratio
        df[thrust_col] *= density_ratio
        df[torque_col] *= density_ratio
        df[power_col] *= density_ratio
        
        # Select the required columns by actual names
        result_df = df[[thrust_col, torque_col, power_col, rpm_col]]
        # Rename them to the expected names for output clarity
        result_df = result_df.rename(columns={
            thrust_col: "Thrust B (N)",
            torque_col: "Torque B (N·m)",
            power_col: "Mechanical Power B (W)",
            rpm_col: "Motor Optical Speed B (RPM)"
        })
        
        # Save the result to a new CSV file in the sealevel_folder
        output_file_path = os.path.join(sealevel_folder, file)
        result_df.to_csv(output_file_path, index=False)

# ------------------ Select files for plotting ------------------------------
plot_files = filedialog.askopenfilenames(
    title="Select performance files AT 0m ISA for separate plots",
    filetypes=[("CSV files", "*.csv")]
)
all_data = pd.concat([pd.read_csv(f) for f in plot_files], ignore_index=True)

if max_rpm == "":
    # compute default max rpm
    max_rpm = 343*0.7 / (np.pi * (float(diameter) * 0.0254)) * 60 if diameter != "" else 343*0.7 / (np.pi * (28 * 0.0254)) * 60
    print(f"Maximum RPM is {max_rpm}")
else:
    max_rpm = float(max_rpm)

# ------------------ PLOTTING and FITS ------------------------------
plt.rcParams.update({'figure.figsize': (10, 7), 'font.size': 9, 'savefig.dpi': 300})

# Thrust vs RPM
plt.figure()
for f in plot_files:
    data = pd.read_csv(f)
    rpm_s = get_series(data, "Motor Optical Speed B (RPM)").abs()
    thrust_s = get_series(data, "Thrust B (N)").abs()
    plt.scatter(rpm_s, thrust_s, label=os.path.basename(f))
    if fit_curve == "Yes":
        x_thrust, y_thrust, OP_thrust, R2_thrust = antfunctions.polyfit_regression(rpm_s, thrust_s, 2, OP, max_rpm)
        
        # Find last negative value in y_thrust and slice arrays accordingly
        y_thrust_array = np.array(y_thrust)
        neg_indices = np.where(y_thrust_array < 0)[0]
        if len(neg_indices) > 0:
            last_neg_idx = neg_indices[-1]
            x_thrust = x_thrust[last_neg_idx:]
            y_thrust = y_thrust[last_neg_idx:]

        plt.plot(x_thrust, y_thrust, label=f"Fitted curve R²={R2_thrust:.2f}")
plt.xlabel('RPM')
plt.ylabel('Thrust B (N)')
plt.title('Thrust vs RPM')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plots_folder, f'{main_name}_Thrust_vs_RPM.png'))

# Torque vs RPM
plt.figure()
for f in plot_files:
    data = pd.read_csv(f)
    rpm_s = get_series(data, "Motor Optical Speed B (RPM)").abs()
    torque_s = get_series(data, "Torque B (N·m)").abs()
    plt.scatter(rpm_s, torque_s, label=os.path.basename(f))
    if fit_curve == "Yes":
        x_torque, y_torque, OP_torque, R2_torque = antfunctions.polyfit_regression(rpm_s, torque_s, 2, OP, max_rpm)

        # remove negative parts if returned by fit
        y_torque_array = np.array(y_torque)
        neg_indices = np.where(y_torque_array < 0)[0]
        if len(neg_indices) > 0:
            last_neg_idx = neg_indices[-1]
            x_torque = x_torque[last_neg_idx:]
            y_torque = y_torque[last_neg_idx:]

        plt.plot(x_torque, y_torque, label=f"Fitted curve R²={R2_torque:.2f}")
plt.xlabel('RPM')
plt.ylabel('Torque (N⋅m)')
plt.title('Torque vs RPM')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plots_folder, f'{main_name}_Torque_vs_RPM.png'))

# Mechanical Power vs Thrust (calculated from fits)
plt.figure()
for f in plot_files:
    data = pd.read_csv(f)
    thrust_s = get_series(data, "Thrust B (N)").abs()
    power_s = get_series(data, "Mechanical Power B (W)").abs()
    rpm_s = get_series(data, "Motor Optical Speed B (RPM)").abs()
    plt.scatter(thrust_s, power_s, label=os.path.basename(f))
    if fit_curve == "Yes":
        x_torque, y_torque, OP_torque, R2_torque = antfunctions.polyfit_regression(rpm_s, get_series(data, "Torque B (N·m)").abs(), 2, OP, max_rpm)    
        x_thrust, y_thrust, OP_thrust, R2_thrust = antfunctions.polyfit_regression(rpm_s, thrust_s, 2, OP, max_rpm)                

        # Align fits by trimming negatives (use thrust negative test)
        y_thrust_array = np.array(y_thrust)
        neg_indices = np.where(y_thrust_array < 0)[0]
        if len(neg_indices) > 0:
            last_neg_idx = neg_indices[-1]
            x_torque = x_torque[last_neg_idx:]
            y_torque = y_torque[last_neg_idx:]
            x_thrust = x_thrust[last_neg_idx:]
            y_thrust = y_thrust[last_neg_idx:]        
        
        y_power = 2*np.pi * x_torque * y_torque / 60
        plt.plot(y_thrust, y_power, label=f"Calculated from fitted thrust and torque")    
plt.xlabel('Thrust B (N)')
plt.ylabel('Mechanical Power B (W)')
plt.title('Mechanical Power vs Thrust')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plots_folder, f'{main_name}_Mechanical_Power_vs_Thrust.png'))

# Thrust vs Torque (from fits)
plt.figure()
for f in plot_files:
    data = pd.read_csv(f)
    torque_s = get_series(data, "Torque B (N·m)").abs()
    thrust_s = get_series(data, "Thrust B (N)").abs()
    plt.scatter(torque_s, thrust_s, label=os.path.basename(f))
    if fit_curve == "Yes":
        x_torque, y_torque, OP_torque, R2_torque = antfunctions.polyfit_regression(get_series(data, "Motor Optical Speed B (RPM)").abs(), torque_s, 2, OP, max_rpm)
        x_thrust, y_thrust, OP_thrust, R2_thrust = antfunctions.polyfit_regression(get_series(data, "Motor Optical Speed B (RPM)").abs(), thrust_s, 2, OP, max_rpm)        
        
        # Trim negatives using thrust
        y_thrust_array = np.array(y_thrust)
        neg_indices = np.where(y_thrust_array < 0)[0]
        if len(neg_indices) > 0:
            last_neg_idx = neg_indices[-1]
            x_torque = x_torque[last_neg_idx:]
            y_torque = y_torque[last_neg_idx:]
            x_thrust = x_thrust[last_neg_idx:]
            y_thrust = y_thrust[last_neg_idx:]
    
        plt.plot(y_torque, y_thrust, label=f"Calculated from fitted thrust and torque")       
plt.xlabel('Torque (N⋅m)')
plt.ylabel('Thrust B (N)')
plt.title('Thrust vs Torque')
plt.legend()
plt.grid()
plt.savefig(os.path.join(plots_folder, f'{main_name}_Thrust_vs_Torque.png'))

# FOM vs Thrust
plt.figure()
for f in plot_files:
    data = pd.read_csv(f)
    thrust_s = get_series(data, "Thrust B (N)").abs()
    power_s = get_series(data, "Mechanical Power B (W)").abs()
    if diameter == "":
        # diameter from file name before first "x"
        diameter = float(os.path.basename(f).split("x")[0])
        FOM = antfunctions.FOM(thrust_s, power_s, diameter, 1.225)
        diameter = ""
    else:
        diameter = float(diameter)
        FOM = antfunctions.FOM(thrust_s, power_s, diameter, 1.225)
    print(f)
    print(f"Diameter: {diameter}")
    plt.plot(thrust_s, FOM, label=os.path.basename(f))
plt.xlabel('Thrust B (N)')
plt.ylabel('Figure of Merit (-)')
plt.title('FOM vs Thrust')
plt.ylim([0,1])
plt.legend()
plt.grid()
plt.savefig(os.path.join(plots_folder, f'{main_name}_FOM_vs_Thrust.png'))

# ------------------ Export table data + coefficients -----------------------
# --- Export block ---
if export_table_data == "Yes":
    for file in os.listdir(FS_avg_folder):
        if file.endswith(".csv"):
            f = os.path.join(FS_avg_folder, file)

            # diameter from filename if not provided
            if diameter == "":
                diameter = float(int(os.path.basename(f).split("x")[0]))
            else:
                diameter = float(diameter)

            data_measured = pd.read_csv(f)

            # Construct path for sea-level corrected file
            sealevel_file_path = os.path.join(sealevel_folder, os.path.basename(f))
            data_sealevel = pd.read_csv(sealevel_file_path)

            # --- Performance coefficients ---
            ct = antfunctions.ct(
                get_column(data_measured, "Thrust B (N)").abs(),
                get_column(data_measured, "Motor Optical Speed B (RPM)").abs(),
                diameter, density
            )
            cp = antfunctions.cp(
                get_column(data_measured, "Mechanical Power B (W)").abs(),
                get_column(data_measured, "Motor Optical Speed B (RPM)").abs(),
                diameter, density
            )
            FOM = antfunctions.FOM(
                get_column(data_measured, "Thrust B (N)").abs(),
                get_column(data_measured, "Mechanical Power B (W)").abs(),
                diameter, density
            )
            if temperature == "":
                tip_speed = antfunctions.tip_speed(
                    get_column(data_measured, "Motor Optical Speed B (RPM)").abs(),
                    diameter
                )
            else:
                tip_speed = antfunctions.tip_speed(
                    get_column(data_measured, "Motor Optical Speed B (RPM)").abs(),
                    diameter, temperature
                )

            # --- Results table ---
            results = pd.DataFrame({
                'RPM [min-1]': get_column(data_measured, "Motor Optical Speed B (RPM)").abs(),
                'Thrust [N]': get_column(data_measured, "Thrust B (N)").abs(),
                'Torque [Nm]': get_column(data_measured, "Torque B (N·m)").abs(),
                'Mechanical power [W]': get_column(data_measured, "Mechanical Power B (W)").abs(),
                'Thrust 0m ISA [N]': get_column(data_sealevel, "Thrust B (N)").abs(),
                'Torque 0m ISA [Nm]': get_column(data_sealevel, "Torque B (N·m)").abs(),
                'Mechanical power 0m ISA [W]': get_column(data_sealevel, "Mechanical Power B (W)").abs(),
                'Ct0 [-]': ct,
                'Cp0 [-]': cp,
                'FoM [-]': FOM,
                'Tip Speed [M]': tip_speed
            })

            # Format floats consistently
            results = results.applymap(lambda x: f"{x:.4f}" if isinstance(x, float) else x)

            # --- Regression Coefficients (use 0m ISA values, no empty rows) ---
            coeffs_tables = []

            rpm_vals = get_column(data_measured, "Motor Optical Speed B (RPM)").abs()

            # Thrust fit (0m ISA)
            thrust_vals = get_column(data_sealevel, "Thrust B (N)").abs()
            x_thrust, y_thrust, OP_thrust, R2_thrust = antfunctions.polyfit_regression(
                rpm_vals, thrust_vals, 2, OP, max_rpm
            )
            thrust_poly = np.polyfit(rpm_vals, thrust_vals, 2)
            coeffs_tables.append(pd.DataFrame({
                "Variable": ["Thrust", "Thrust", "Thrust"],
                "Polynomial": ["T1", "T2", "T3"],
                "Coeff": [f"{c:.2E}" for c in thrust_poly],
                "R2": [f"{R2_thrust:.3f}", "", ""]
            }))

            # Torque fit (0m ISA)
            torque_vals = get_column(data_sealevel, "Torque B (N·m)").abs()
            x_torque, y_torque, OP_torque, R2_torque = antfunctions.polyfit_regression(
                rpm_vals, torque_vals, 2, OP, max_rpm
            )
            torque_poly = np.polyfit(rpm_vals, torque_vals, 2)
            coeffs_tables.append(pd.DataFrame({
                "Variable": ["Torque", "Torque", "Torque"],
                "Polynomial": ["Q1", "Q2", "Q3"],
                "Coeff": [f"{c:.2E}" for c in torque_poly],
                "R2": [f"{R2_torque:.3f}", "", ""]
            }))

            # Power fit (0m ISA, cubic)
            power_vals = get_column(data_sealevel, "Mechanical Power B (W)").abs()
            power_poly = np.polyfit(rpm_vals, power_vals, 3)
            y_pred = np.polyval(power_poly, rpm_vals)
            ss_res = np.sum((power_vals - y_pred) ** 2)
            ss_tot = np.sum((power_vals - np.mean(power_vals)) ** 2)
            R2_power = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

            coeffs_tables.append(pd.DataFrame({
                "Variable": ["Mech Power", "Mech Power", "Mech Power", "Mech Power"],
                "Polynomial": ["P1", "P2", "P3", "P4"],
                "Coeff": [f"{c:.2E}" for c in power_poly],
                "R2": [f"{R2_power:.3f}", "", "", ""]
            }))

            coeffs_df = pd.concat(coeffs_tables, ignore_index=True)

            # --- Write output CSV ---
            output_file_path = os.path.join(FS_data, f"data-table_{os.path.basename(f)}")
            results.to_csv(output_file_path, index=False)  # overwrite
            with open(output_file_path, "a") as out_f:     # append
                out_f.write("\n\n")  # blank line before coefficients
                coeffs_df.to_csv(out_f, index=False)

# ------------------ Export coefficients (extrapolated) ---------------------
if export_extrapolated_data == "Yes":
    for file in os.listdir(FS_avg_folder):
        if file.endswith(".csv"):
            f = os.path.join(FS_avg_folder, file)

            data = pd.read_csv(f)
            if diameter == "":
                diameter = float(int(os.path.basename(f).split("x")[0]))
            else:
                diameter = float(diameter)

            # Get series via alias
            rpm_s = get_series(data, "Motor Optical Speed B (RPM)").abs()
            thrust_s = get_series(data, "Thrust B (N)").abs()
            torque_s = get_series(data, "Torque B (N·m)").abs()
            power_s = get_series(data, "Mechanical Power B (W)").abs()

            # Use your existing polyfit_regression to create fitted arrays (for plotting/extrapolation)
            x_torque, y_torque, OP_torque, R2_torque = antfunctions.polyfit_regression(rpm_s, torque_s, 2, OP, max_rpm)
            x_thrust, y_thrust, OP_thrust, R2_thrust = antfunctions.polyfit_regression(rpm_s, thrust_s, 2, OP, max_rpm)   
            y_power_from_fits = None
            try:
                # Fit polynomial to torque and thrust directly to build power via 2*pi * Q * rpm / 60
                # But also compute polynomial fit for power itself (cubic)
                power_poly = np.polyfit(rpm_s, power_s, 3)
                # compute R2 for power
                y_pred_power = np.polyval(power_poly, rpm_s)
                ss_res = np.sum((power_s - y_pred_power)**2)
                ss_tot = np.sum((power_s - np.mean(power_s))**2)
                R2_power = 1.0 - ss_res/ss_tot if ss_tot != 0 else 1.0
            except Exception:
                power_poly = np.array([0,0,0,0])
                R2_power = 0.0

            # Prepare DataFrame of extrapolated x/y (from polyfit_regression)
            # Use thrust-based x,y and torque-based y for torque
            df_extrap = pd.DataFrame({
                'RPM': x_thrust,
                'Thrust [N]': y_thrust,
                'Torque [Nm]': y_torque,
            })
            # If torque polynomial returned different x than thrust, attempt to align:
            if len(x_torque) == len(x_thrust):
                df_extrap['Torque [Nm]'] = y_torque
            else:
                # Re-evaluate torque polynomial at x_thrust if torque coefficients are available
                try:
                    torque_coef = np.polyfit(rpm_s, torque_s, 2)
                    df_extrap['Torque [Nm]'] = np.polyval(torque_coef, x_thrust)
                except Exception:
                    df_extrap['Torque [Nm]'] = np.nan

            # mechanical power from torque poly (if possible)
            try:
                # If torque coefficients available as above
                torque_coef = np.polyfit(rpm_s, torque_s, 2)
                torque_at_x = np.polyval(torque_coef, df_extrap['RPM'])
                df_extrap['Mechanical Power [W]'] = 2*np.pi * df_extrap['RPM'] * torque_at_x / 60
            except Exception:
                # fallback to power poly directly
                df_extrap['Mechanical Power [W]'] = np.polyval(power_poly, df_extrap['RPM'])

            # Find the last negative value in 'Thrust [N]'
            last_neg_index = df_extrap[df_extrap['Thrust [N]'] < 0].index.max()
            if last_neg_index is not None:
                df_extrap = df_extrap.iloc[last_neg_index+1:]  # Keep rows after the last negative thrust value

            # Save the extrapolated data CSV
            new_file = os.path.join(FS_data, f"extrapolated_{os.path.basename(f)}")
            df_extrap.to_csv(new_file, index=False)

# ------------------ Final script version reporting -----------------------
antfunctions.script_version(os.path.dirname(main))
