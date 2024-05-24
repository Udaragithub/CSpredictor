import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from tensorflow.keras.models import load_model
import tkinter as tk
from tkinter import Label, Entry, Button, messagebox, Text, END
import matplotlib.pyplot as plt

# Register 'mse' as Mean Squared Error
from tensorflow.keras.utils import get_custom_objects
get_custom_objects().update({'mse': tf.keras.losses.MeanSquaredError()})

# Load the LCA data
try:
    lca_df = pd.read_csv("Carbon_db.csv")
    print("LCA data loaded successfully.")
except Exception as e:
    print(f"Error loading LCA data: {e}")

# Load the trained DNN model and scaler
try:
    scaler = joblib.load('scaler.pkl')
    dnn_model = load_model('dnn_model.h5')
    print("Model and scaler loaded successfully.")
except Exception as e:
    print(f"Error loading model or scaler: {e}")

def gwp(var1, var2, var3, var4, var5, var6, var7, var8):
    try:
        var1, var2, var3, var4, var5, var6, var7, var8 = map(float, [var1, var2, var3, var4, var5, var6, var7, var8])
        plc_gwp = var1 * float(lca_df.loc[lca_df['Element'] == 'PLC', 'GWP'].iloc[0])
        ggbfs_gwp = var2 * float(lca_df.loc[lca_df['Element'] == 'GGBFS', 'GWP'].iloc[0])
        cca_gwp = var3 * float(lca_df.loc[lca_df['Element'] == 'CCA', 'GWP'].iloc[0])
        fa_gwp = var4 * float(lca_df.loc[lca_df['Element'] == 'FA', 'GWP'].iloc[0])
        ca_gwp = var5 * float(lca_df.loc[lca_df['Element'] == 'CA', 'GWP'].iloc[0])
        sh_gwp = var6 * float(lca_df.loc[lca_df['Element'] == 'SH', 'GWP'].iloc[0])
        ss_gwp = var7 * float(lca_df.loc[lca_df['Element'] == 'SS', 'GWP'].iloc[0])
        w_gwp = var8 * float(lca_df.loc[lca_df['Element'] == 'W', 'GWP'].iloc[0])

        tot_gwp = plc_gwp + ggbfs_gwp + cca_gwp + fa_gwp + ca_gwp + sh_gwp + ss_gwp + w_gwp

        return tot_gwp

    except ValueError:
        raise ValueError("All input variables must be numeric")

def generate_random_data(number_of_gen_data):
    min_max_vals = {
        "GGBFS": {"min": 0, "max": 488},
        "CCA": {"min": 0, "max": 488},
        "FA": {"min": 728, "max": 899},
        "W": {"min": 32.64, "max": 37.86},
        "SHP": {"min": 20.74, "max": 25.96},
        "CD": {"min": 7, "max": 90},
        "MC": {"min": 12, "max": 16},
        "CG": {"min": 30, "max": 40},
    }

    gen_data = np.random.rand(number_of_gen_data, 8)
    count = 0
    for i in min_max_vals.keys():
        min_ = min_max_vals[i]["min"]
        max_ = min_max_vals[i]["max"]
        gen_data[:, count] = (gen_data[:, count] * (max_ - min_)) + min_
        count += 1

    df = pd.DataFrame(gen_data, columns=min_max_vals.keys())
    return df

def get_final_variables(df, expected_value, max_error_allowed):
    df["prediction"] = dnn_model.predict(scaler.transform(df))
    df["prediction"] = df["prediction"].astype(float)
    df["error"] = df["prediction"] - expected_value
    df["abs_error"] = abs(df["error"])
    df["abs_error_%"] = df["abs_error"] * 100 / abs(expected_value)
    df = df.sort_values(by=["abs_error_%"])
    final_variables = df[df["abs_error_%"] < max_error_allowed].reset_index(drop=True)

    final_variables['GWP(kgCO2eq/m3)'] = final_variables['GGBFS'] * float(
        lca_df.loc[lca_df['Element'] == 'GGBFS', 'GWP'].iloc[0]) + final_variables['CCA'] * float(
        lca_df.loc[lca_df['Element'] == 'CCA', 'GWP'].iloc[0]) + final_variables['FA'] * float(
        lca_df.loc[lca_df['Element'] == 'FA', 'GWP'].iloc[0]) + final_variables['W'] * float(
        lca_df.loc[lca_df['Element'] == 'W', 'GWP'].iloc[0]) + final_variables['SHP'] * float(
        lca_df.loc[lca_df['Element'] == 'SHP', 'GWP'].iloc[0])
    final_variables.insert(0, 'case_name', ['case-{}'.format(i + 1) for i in range(len(final_variables))])

    final_variables = final_variables.sort_values(by='GWP(kgCO2eq/m3)', ascending=True).reset_index(drop=True)
    final_variables = final_variables.round(1)

    return final_variables

def on_submit():
    try:
        expected_value = float(entry.get())
        max_error_allowed = float(entry_max_error_allowed.get())
        number_of_gen_data = 1000
        attempts = 0
        max_attempts = 10

        final_vars = pd.DataFrame()
        while len(final_vars) < 5 and attempts < max_attempts:
            df = generate_random_data(number_of_gen_data)
            final_vars = get_final_variables(df, expected_value, max_error_allowed)
            attempts += 1

        result_text.delete(1.0, END)

        if len(final_vars) >= 5:
            print(final_vars)
            result_text.insert(END, final_vars.head(10).to_string(index=False))

            plt.bar(final_vars['case_name'].head(10), final_vars['GWP(kgCO2eq/m3)'].head(10), color='green')
            plt.xlabel('Case Name')
            plt.ylabel('GWP(kgCO2eq/m3)')
            plt.title('Global Warming Potential of Concrete Mixes')
            plt.xticks(rotation=45, ha='right')
            plt.ylim(15, max(final_vars['GWP(kgCO2eq/m3)'].max(), 15))
            plt.show()
        else:
            error_message = "No solutions found with less than the maximum allowed error after 10 attempts."
            print(error_message)
            result_text.insert(END, error_message)

    except ValueError as e:
        messagebox.showerror("Error", f"Error: {str(e)}")
        print(f"Error: {str(e)}")

def on_exit():
    root.destroy()

root = tk.Tk()
root.title("Slag-Ash-Based Geopolymer Concrete Mix Design Calculator")

Label(root, text="Expected Compressive Strength (MPa):").pack(pady=10)
entry = Entry(root)
entry.pack(pady=10)

Label(root, text="Maximum Error Allowed(%):").pack(pady=10)
entry_max_error_allowed = Entry(root)
entry_max_error_allowed.pack(pady=10)

Button(root, text="Calculate Trial Mix", command=on_submit).pack(pady=10)
Button(root, text="Exit", command=on_exit).pack(pady=10)

result_text = Text(root, height=10, width=150)
result_text.pack(pady=10)

root.mainloop()
