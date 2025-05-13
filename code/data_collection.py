import logging
import os
import random
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from bs4 import BeautifulSoup
from tqdm import tqdm


def fetch_news(search: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetches news articles from Google News for a given search term within a specified date range.
    This function scrapes news articles from Google News by iterating through each day in the
    specified date range. It collects the article titles, publication dates, and links, and
    returns the data as a pandas DataFrame.
    Args:
        search (str): The search term to query on Google News.
        start_date (str): The start date of the date range in the format 'YYYY-MM-DD'.
        end_date (str): The end date of the date range in the format 'YYYY-MM-DD'.
    Returns:
        pandas.DataFrame: A DataFrame containing the fetched news articles with the following columns:
            - 'date': The publication date of the news article.
            - 'title': The title of the news article.
            - 'link': The URL link to the news article.
    Raises:
        Exception: Logs an error message if there is an issue fetching data for a specific date.
    Notes:
        - The function uses the `requests` library to fetch the HTML content of the Google News search results.
        - The `BeautifulSoup` library is used to parse the HTML and extract relevant data.
        - A progress bar is displayed using the `tqdm` library to indicate the progress of the data fetching process.
        - A delay of 1 second is added between requests to avoid overwhelming the server.
    """

    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    current = start
    runs = (end - start).days
    df = pd.DataFrame(columns=["title", "date", "link"])

    logging.info(f"Fetching news articles for '{search}' from {start_date} to {end_date}...")
    pbar = tqdm(total=runs + 1)
    while current <= end:
        current_date = current.strftime("%Y-%m-%d")
        next_date = (current + timedelta(days=1)).strftime("%Y-%m-%d")
        url = f"https://news.google.com/search?q={search}%20after%3A{current_date}%20before%3A{next_date}&hl=en-US&gl=US&ceid=US%3Aen"
        logging.info(f"Fetching data for date: {current_date}")
        try:
            response = requests.get(url)
            response.raise_for_status()
            soup = BeautifulSoup(
                response.text,
                "html.parser",
            )

            temp = soup.find_all("a", {"class": "JtKRv"})
            times = soup.find_all("time", {"class": "hvbAAd"})
            data = []
            for i in range(len(temp)):
                link = "https://news.google.com/" + temp[i]["href"][2:]
                title = temp[i].text
                date = times[i]["datetime"]
                data.append((date, title, link))

            current_data = pd.DataFrame(data, columns=["date", "title", "link"])
            current_data = current_data.sort_values(by=["date", "title"], ascending=[True, True])

            df = pd.concat([df, current_data], ignore_index=True)
            if len(current_data) > 0:
                logging.info(f"Fetched {len(current_data)} articles for date: {current_date}")
            else:
                logging.warning(f"No articles found for date: {current_date}")
        except Exception as e:
            logging.error(f"Error fetching data for date {current_date}: {e}")
        finally:
            current += timedelta(days=1)
            time.sleep(random.uniform(1, 4))  # Random sleep between 1 and 2 seconds
            pbar.update(1)

    logging.info("Data fetching complete.")
    return df


if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        filename="data_collection.log",
        filemode="a",
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
        # handlers=[
        #     logging.FileHandler("data_collection.log"),
        #     logging.StreamHandler()
        # ]
    )

    commodity_list = ["wheat"]  # "canola", "corn", "ethanol", "gasoline", "oats", "soybean", "sugarcane",
    start_date = "2015-01-01"
    end_date = "2025-04-30"

    for search in commodity_list:
        try:
            print(f"Fetching news for {search} from {start_date} to {end_date}")
            df = fetch_news(search, start_date, end_date)
            # Remove duplicates based on all columns (title, date, and link)
            df = df.drop_duplicates(subset=["title", "date", "link"], keep="first")
            if df.empty:
                logging.info("No data fetched.")
            else:
                logging.info(f"Fetched {len(df)} articles.")

            # Save the DataFrame to a CSV file
            # Ensure the directory exists
            os.makedirs("../data/news", exist_ok=True)
            output_file = f"../data/news/news_{search}_{start_date}_to_{end_date}.csv"
            df.to_csv(output_file, index=False)
            logging.info(f"Data saved to {output_file}")
        except Exception as e:
            logging.error(f"An error occurred: {e}")
