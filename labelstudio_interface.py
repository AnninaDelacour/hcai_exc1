import requests
import os
from dotenv import load_dotenv
from active_learning_basics import already_labeled

load_dotenv()

LS_URL = "http://10.203.222.235:8080"
LS_PROJECT_ID = 11
LS_API_TOKEN = os.getenv("API_TOKEN")


def get_access_token():
    response = requests.post(
        f"{LS_URL}/api/token/refresh",
        headers={"Content-Type": "application/json"},
        json={"refresh": LS_API_TOKEN}
    )
    if response.status_code != 200:
        raise Exception(f"Token refresh failed: {response.text}")
    return response.json()["access"]


def get_headers():
    return {
        "Authorization": f"Bearer {get_access_token()}",
        "Content-Type": "application/json"
    }


def convert_to_labelstudio_format(items):
    return [{
        "data": {"text": item[1]},
          "meta": {
              "id": item[0],
              "strategy": item[3],
              "confidence": item[4]
              }
            } for item in items]


def upload_to_labelstudio(items):
    tasks = convert_to_labelstudio_format(items)
    response = requests.post(
        f"{LS_URL}/api/projects/{LS_PROJECT_ID}/import",
        headers=get_headers(), 
        json=tasks
    )
    if response.status_code == 201:
        print(f"Successfully imported {len(tasks)} tasks.")
    else:
        print(f"Error during import: {response.status_code}")
        print(response.text)



def fetch_labeled_data(original_items):
    """
    Gets data from Label Studio which matches with the original ID
    """
    response = requests.get(
        f"{LS_URL}/api/projects/{LS_PROJECT_ID}/tasks/?completed=true",
        headers=get_headers()
    )

    if response.status_code != 200:
        print(f"Error during call of labelled data: {response.status_code}")
        return original_items

    label_data = response.json()
    label_map = {}

    for task in label_data:
        task_meta_id = task.get('meta', {}).get('id')
        if task_meta_id is not None:
            task_id_str = str(task_meta_id)
        else:
            continue  # ignore if without meta-id

        try:
            label_value = task['annotations'][0]['result'][0]['value']['choices'][0]
            label_map[task_id_str] = label_value
        except (IndexError, KeyError):
            continue  # skip if incomplete

    updated_items = []
    for item in original_items:
        item_id = str(item[0])  # make sure both are strings
        if item_id in label_map:
            label_str = label_map[item_id].lower()
            if label_str in ["positive", "neutral", "negative"]:
                item[2] = label_str
                updated_items.append(item)
            else:
                print(f"Ignore unknown label: {label_str}")
        else:
            print(f"No more annotation found for Item ID {item_id}.")
    
    # update already_labeled variable
    for item in updated_items:
        item_id = str(item[0])
        item_label = item[2]
        already_labeled[item_id] = item_label

    return updated_items

