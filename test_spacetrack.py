import os
import logging
logging.basicConfig(level=logging.DEBUG)

# Mock Celestrak URL to force failure
import src.tle_fetcher
src.tle_fetcher.CELESTRAK_GP_URL = "http://localhost:9999/dummy"

os.environ["SPACETRACK_USERNAME"] = "dummy_user"
os.environ["SPACETRACK_PASSWORD"] = "dummy_pass"

print("Testing fetch fallback...")
res = src.tle_fetcher.fetch_tle("25544", max_retries=0)
print(f"Result: {res}")
