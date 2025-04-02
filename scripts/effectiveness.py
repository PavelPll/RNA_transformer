# Find optimal number of pages in URL
# for the fastest data extraction
# from rnacentral
# page=50 seems to be the best choice

import requests
import time

# Base URL for the API
url_template = "https://rnacentral.org/api/v1/rna/?page={}&page_size={}"

# List of page sizes to test
page_sizes = [1, 10, 25, 30, 40, 45, 50, 55, 60, 70, 75, 100, 150, 200]
page_sizes = [46, 47, 48, 49, 50, 51, 52, 53, 54]

# Number of test runs per page_size
num_requests = 10

def test_page_size(page_size):
    total_time = 0
    total_entries = 0

    for i in range(num_requests):
        url = url_template.format(i + 1, page_size)  # Fetch different pages
        start_time = time.time()

        response = requests.get(url)
        elapsed_time = time.time() - start_time

        if response.status_code == 200:
            page = response.json()
            for el in page['results']:
                sequence = el["sequence"]
            num_entries = len(page['results'])  # Extract number of records

            total_entries += num_entries
            total_time += elapsed_time
        else:
            print(f"Error with page_size={page_size}, status code: {response.status_code}")
            return None  # Skip this page_size if it fails

    # Calculate efficiency: entries retrieved per second
    efficiency = total_entries / total_time if total_time > 0 else 0

    return efficiency

# Run the test for each page_size
results = {}
for size in page_sizes:
    efficiency = test_page_size(size)
    if efficiency is not None:
        results[size] = efficiency
        print(f"✅ Page Size {size}: {efficiency:.2f} entries/sec")

# Find the optimal page_size
if results:
    best_page_size = max(results, key=results.get)
    print(f"\n🚀 Optimal page_size: {best_page_size} with {results[best_page_size]:.2f} entries/sec")

"""✅ Page Size 1: 1.25 entries/sec
✅ Page Size 10: 8.56 entries/sec
✅ Page Size 25: 14.81 entries/sec
✅ Page Size 30: 8.09 entries/sec
✅ Page Size 40: 19.34 entries/sec
✅ Page Size 45: 9.83 entries/sec
✅ Page Size 50: 482.16 entries/sec
✅ Page Size 55: 10.14 entries/sec
✅ Page Size 60: 21.05 entries/sec
✅ Page Size 70: 10.81 entries/sec
✅ Page Size 75: 23.10 entries/sec
✅ Page Size 100: 25.33 entries/sec
✅ Page Size 150: 26.11 entries/sec
✅ Page Size 200: 27.40 entries/sec"""