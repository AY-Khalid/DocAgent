import os
import requests
from bs4 import BeautifulSoup
from urllib.parse import quote

# ✅ Curated list of 30 common ailments with valid WHO URL slugs
ailments = [
    "coronavirus-disease-(covid-19)",
    "diabetes",
    "hypertension",
    "depression",
    "tuberculosis",
    "malaria",
    "hiv-aids",
    "cancer",
    "asthma",
    "dengue-and-severe-dengue",
    "hepatitis-b",
    "hepatitis-c",
    "stroke",
    "mental-health",
    "pneumonia-in-children",
    "measles",
    "cholera",
    "meningitis",
    "epilepsy",
    "breast-cancer",
    "obesity-and-overweight",
    "anxiety-disorders",
    "influenza-(seasonal)",
    "alcohol",
    "smoking-and-tobacco",
    "oral-health",
    "arthritis",
    "skin-conditions",
    "hearing-loss",
    "vision-impairment-and-blindness"
]

# WHO base URL and output folder
base_url = "https://www.who.int/news-room/fact-sheets/detail/"
output_dir = "data/corpus/"
os.makedirs(output_dir, exist_ok=True)

# Loop to scrape and save each ailment's text content
for slug in ailments:
    url = base_url + quote(slug)
    try:
        res = requests.get(url)
        res.raise_for_status()
        soup = BeautifulSoup(res.text, 'html.parser')

        # Use the correct container div by ID (as you noted)
        content = soup.find("div", id="PageContent_T0643CD2A003_Col00")

        if not content:
            print(f"⚠️ No content found for: {slug}")
            continue

        text = content.get_text(separator="\n", strip=True)
        filename = f"{slug}.txt"
        filepath = os.path.join(output_dir, filename)

        with open(filepath, "w", encoding="utf-8") as f:
            f.write(text)

        print(f"✅ Saved: {filename}")

    except Exception as e:
        print(f"❌ Failed to scrape {slug}: {e}")
