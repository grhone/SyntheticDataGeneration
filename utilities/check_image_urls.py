"""
Image URL Validator Module

This module provides functionality to validate image URLs/paths in JSON data containing chat messages.
It checks whether referenced image files exist on the local filesystem.

Functions:
    validate_image_urls(json_data: list) -> list:
        Scans JSON data structure for image URLs and verifies their existence.
        Returns a list of dictionaries containing invalid image paths and context.

Usage:
    Run as a script from command line:
        python check_image_urls.py <path_to_json_file>
    
    Returns:
        - List of invalid image paths with context if any are found
        - Success message if all paths are valid
        - Error messages for file/JSON parsing issues

Example JSON Structure Expected:
    [
        {
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": {"url": "path/to/image.jpg"}
                        }
                    ]
                },
                {
                    "role": "assistant",
                    "content": "..."  # Context for invalid images
                }
            ]
        }
    ]

Dependencies:
    - colorama (for colored terminal output)
    - Python standard libraries: os, json, argparse
"""

import os
import json
import argparse
from colorama import Fore, init 
from utilities.logger import setup_logger

# Initialize colorama
init(autoreset=True)

# Initialize logging 
logger = setup_logger(__name__)

def validate_image_urls(json_data):
    invalid_images = []
    
    for item in json_data:
        for message in item["messages"]:
            if message["role"] == "user":
                for content in message["content"]:
                    if content["type"] == "image_url":
                        image_path = content["image_url"]["url"]
                        if not os.path.exists(image_path):
                            invalid_images.append({
                                "image_url": image_path,
                                "context": item["messages"][1]["content"][:100]  # assistant's response for context
                            })
    return invalid_images

def check_image_urls_in_file(file_path: str):
    """Check image URLs in a JSON file and report invalid ones"""
    try:
        with open(file_path, "r") as f:
            data = json.load(f)
    
        invalid = validate_image_urls(data)
        
        if invalid:
            print(f"Found {len(invalid)} invalid image path(s) in {file_path}:")
            for img in invalid:
                print(f"\n- Image path: {img['image_url']}")
                print(f"  Context: {img['context']}")
            return False
        else:
            print(f"All image paths in {file_path} are valid!")
            return True
    except Exception as e:
        print(f"Error checking {file_path}: {str(e)}")
        return False

def check_all_output_files(output_dir: str):
    """Check image URLs in all JSON files in output directory"""
    all_valid = True
    for filename in os.listdir(output_dir):
        if filename.endswith('.json'):
            file_path = os.path.join(output_dir, filename)
            if not check_image_urls_in_file(file_path):
                all_valid = False
    return all_valid

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate image URLs in JSON data')
    parser.add_argument('json_file', help='Path to JSON file containing image URLs')
    args = parser.parse_args()

    try:
        with open(args.json_file, "r") as f:
            data = json.load(f)
    
        invalid = validate_image_urls(data)
        
        if invalid:
            print(f"Found {len(invalid)} invalid image path(s) in {args.json_file}:")
            for img in invalid:
                print(f"\n{Fore.RED}- Image path: {img['image_url']}")
                print(f"  Context: {img['context']}")
        else:
                print(f"All image paths in {args.json_file} are valid!")
    except FileNotFoundError:
        print(f"Error: File '{args.json_file}' not found")
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON in '{args.json_file}'")
