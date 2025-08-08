import os, yaml
from dotenv import load_dotenv

def load_settings():
    load_dotenv()
    with open("config/settings.yaml", "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    cfg["env"] = {
        "GROQ_API_KEY": os.getenv("GROQ_API_KEY", ""),
    }
    return cfg


