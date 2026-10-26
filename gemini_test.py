import os
import time

from google import genai

client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

# Κρατάμε μόνο τα γενικά μοντέλα κειμένου (όχι φωνή, ήχο, live, εικόνα)
SKIP = ("tts", "audio", "live", "image", "native", "embed")
names = sorted(
    m.name.replace("models/", "")
    for m in client.models.list()
    if "flash" in m.name and not any(s in m.name for s in SKIP)
)
print("Δοκιμάζω:", names, "\n")

for name in names:
    try:
        r = client.models.generate_content(
            model=name,
            contents="Απάντησε με μία πρόταση στα ελληνικά: τι είναι μια δημόσια διαβούλευση;",
        )
        print(f"ΟΚ   {name}: {r.text.strip()[:100]}")
    except Exception as e:
        print(f"ΣΦΑΛΜΑ {name}: {str(e)[:150]}")
    time.sleep(3)