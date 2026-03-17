import requests
url = "https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=json"
resp = requests.get(url)
print(resp.status_code)
data = resp.json()
print("Count:", len(data))
if len(data) > 0:
    for k, v in data[0].items():
        print(f"{k}: {v}")
