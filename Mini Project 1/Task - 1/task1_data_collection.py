import requests
import time
import json
import os
from datetime import datetime

#my categories
categories = {
    "technology": ["AI", "software", "tech", "code", "computer", "data", "cloud", "API", "GPU", "LLM"],
    "worldnews": ["war", "government", "country", "president", "election", "climate", "attack", "global"],
    "sports": ["NFL", "NBA", "FIFA", "sport", "game", "team", "player", "league", "championship"],
    "science": ["research", "study", "space", "physics", "biology", "discovery", "NASA", "genome"],
    "entertainment": ["movie", "film", "music", "Netflix", "game", "book", "show", "award", "streaming"]
}

#by using this base url, and endpoints we can get data
BASE_URL = "https://hacker-news.firebaseio.com/v0"
headers = {"User-Agent": "TrendPulse/1.0"}
print("Fetching top story IDs...")
try:
    response = requests.get(f"{BASE_URL}/topstories.json", headers=headers)
    top_ids = response.json()[:200]
    print(f"Got {len(top_ids)} story IDs")
except Exception as e:
    print(f"Failed to fetch story IDs: {e}")
    top_ids = []

all_stories = []
#accessing each category to check with that each story title
for category, keywords in categories.items():
    print(f"\nCollecting stories for: {category}")
    count = 0
    for story_id in top_ids:
        if count >= 25: 
            break

        try:
            story_response = requests.get(f"{BASE_URL}/item/{story_id}.json", headers=headers)
            story = story_response.json()
            #checking if title matches o not
            if not story or "title" not in story:
                continue

            title = story.get("title", "")
            #checking each keyword to match with title
            for keyword in keywords:
                if keyword.lower() in title.lower():
                    all_stories.append({
                        "post_id": story.get("id"),
                        "title": title,
                        "category": category,
                        "score": story.get("score"),
                        "num_comments": story.get("descendants"),
                        "author": story.get("by"),
                        "collected_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })
                    count += 1
                    break

        except Exception as e:
            print(f"Failed to fetch story {story_id}: {e}")
            continue

    time.sleep(2)

os.makedirs("data", exist_ok=True)
#mking new json file..
date_str = datetime.now().strftime("%Y%m%d")
filename = f"data/trends_{date_str}.json"

with open(filename, "w") as f:
    json.dump(all_stories, f, indent=4)

print(f"Collected {len(all_stories)} stories. Saved to {filename}")