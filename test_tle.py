import requests
resp = requests.get('https://celestrak.org/NORAD/elements/gp.php?GROUP=cosmos-2251-debris&FORMAT=TLE')
lines = [L.strip() for L in resp.text.splitlines() if L.strip()]
print(lines[:6])
