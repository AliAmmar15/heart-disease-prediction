import requests
import pandas as pd
import os
import matplotlib.pyplot as plt

def save_plot(fig, filename):
    """Saves a matplotlib figure to the results folder."""
    os.makedirs("results", exist_ok=True)
    fig.savefig(f"results/{filename}", bbox_inches="tight", dpi=300)
    plt.close(fig)

def generate_report(model_name, classification_report):
    """Formats a classification report into Markdown."""
    return f"# {model_name} Classification Report\n```\n{classification_report}\n```"

def fetch_cdc_data():
    """Fetches heart disease death rates per state from CDC API."""
    url = "https://data.cdc.gov/resource/3yf8-kanr.json"  
    try:
        response = requests.get(url)
        data = response.json()
        print("\n CDC API Response (First 5 records):", data[:5])  

        df = pd.DataFrame(data)

        
        if "jurisdiction_of_occurrence" not in df.columns or "diseases_of_heart_i00_i09" not in df.columns:
            print("⚠ CDC API response format has changed. Skipping CDC data.")
            return None

        
        df["diseases_of_heart_i00_i09"] = pd.to_numeric(df["diseases_of_heart_i00_i09"], errors="coerce")

        
        df = df.dropna()

        
        df.rename(columns={"diseases_of_heart_i00_i09": "heart_disease_rate"}, inplace=True)

        return df
    except Exception as e:
        print(f" Error fetching CDC data: {e}")
        return None

def fetch_fda_data():
    """Fetches drug reaction data from OpenFDA API."""
    url = "https://api.fda.gov/drug/event.json?limit=10"
    try:
        response = requests.get(url)
        data = response.json()
        results = data.get("results", [])

        extracted_data = []
        for record in results:
            patient = record.get("patient", {}).get("reaction", [])
            for reaction in patient:
                extracted_data.append({
                    "reaction": reaction.get("reactionmeddrapt", ""),
                    "seriousness": record.get("serious", 0)
                })

        df = pd.DataFrame(extracted_data)

        
        df = df.dropna()

        
        if not df.empty and "reaction" in df.columns:
            df = pd.get_dummies(df, columns=["reaction"], prefix="rx")

        return df
    except Exception as e:
        print(f" Error fetching FDA data: {e}")
        return None
