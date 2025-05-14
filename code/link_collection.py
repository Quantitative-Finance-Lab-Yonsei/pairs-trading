import os
import random
import time

import pandas as pd
from selenium import webdriver
from selenium.common.exceptions import WebDriverException
from tqdm import tqdm


def get_redirect_link(url):
    """
    Resolves a redirect URL to its final destination URL using Selenium.

    Args:
      url: The initial URL that may be a redirect.

    Returns:
      The final destination URL after following redirects, or the original URL
      if an error occurred.
    """
    # Set up the WebDriver (e.g., ChromeDriver). Make sure the driver executable
    # is in your system's PATH or provide the path to the executable.
    # You might need to change this based on the browser you have installed (e.g., Firefox, Edge).
    driver = None  # Initialize driver to None
    try:
        # Using Chrome as an example. You might need options like --headless
        options = webdriver.ChromeOptions()
        # if you don't want a browser window to open.
        # Uncomment the line below to run in headless mode (no browser window)
        options.add_argument("--headless")
        # options.add_argument('--no-sandbox') # Recommended for some environments
        # options.add_argument('--disable-dev-shm-usage') # Recommended for some environments

        driver = webdriver.Chrome(options=options)
        # print(f"Attempting to resolve URL: {url}")
        driver.get(url)

        # Wait for the body element to be present. This is a more reliable indicator
        # that the page has loaded after redirects compared to checking the title.
        # print("sleeping for 1 seconds to allow for any additional redirects...")
        time.sleep(random.uniform(2, 5))  # Sleep for a random time between 1 and 2 seconds
        # Stop the driver from loading further by executing a script to stop network activity
        # print("Stopping the driver from loading further...")
        driver.execute_script("window.stop();")
        # Get the current URL after all redirects have occurred
        final_url = driver.current_url
        # print(f"Final URL resolved by Selenium: {final_url}")

        return final_url

    except WebDriverException:
        # print(f"An error occurred with Selenium: {e}")
        return url  # Return original URL if an error occurs
    finally:
        # Always close the browser session if the driver was successfully initialized
        if driver:
            driver.quit()
            # print("Browser session closed.")


data_directory = "../data/news/"
news_data = [x for x in os.listdir(data_directory) if x.endswith(".csv")]

for file in news_data:
    file_path = os.path.join(data_directory, file)
    save_path = os.path.join("../data/news_sentiment", file)
    commodity = file.split("_")[1]
    print(f"Processing file: {file_path} for commodity: {commodity}")
    df = pd.read_csv(file_path)
    df["final_url"] = None
    for index, row in tqdm(df.iterrows(), total=len(df)):
        df.at[index, "final_url"] = get_redirect_link(row["link"])
        print(f"Final URL for {row['title']}: {df.at[index, 'final_url']}")
        if "sorry" in df.at[index, "final_url"]:
            print(f"Sorry page encountered for URL: {row['link']}")
            break
    df.to_csv(file_path, index=False)
