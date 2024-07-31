# import httpx
# from configparser import ConfigParser
#
# config = ConfigParser()
# config.read('credentials.ini')
# api_key = config['BingAPI']['api_key']
#
# web_search_endpoint = ""


#Copyright (c) Microsoft Corporation. All rights reserved.
#Licensed under the MIT License.

# -*- coding: utf-8 -*-

import json
import os
from pprint import pprint
import requests
from dotenv import load_dotenv

load_dotenv()

'''
This sample makes a call to the Bing Web Search API with a query and returns relevant web search.
Documentation: https://docs.microsoft.com/en-us/bing/search-apis/bing-web-search/overview
'''

# Add your Bing Search V7 subscription key and endpoint to your environment variables.
subscription_key = os.getenv('BING_SEARCH_V7_SUBSCRIPTION_KEY')
endpoint = os.getenv('BING_SEARCH_V7_ENDPOINT') + "v7.0/search"


def search(query):
    # Query term(s) to search for.
    # query = "股市"

    # Construct a request
    mkt = 'zh-CN'
    params = {'q': query, 'mkt': mkt}
    headers = {'Ocp-Apim-Subscription-Key': subscription_key}

    # Call the API
    try:
        response = requests.get(endpoint, headers=headers, params=params)
        response.raise_for_status()

        # print("Headers:")
        # print(response.headers)
        #
        # print("JSON Response:")
        # pprint(response.json())
        return response.json()['webPages']['value']
    except Exception as ex:
        raise ex


results = search("昨天发生了什么事")

results_prompts = [
    f"Source:\nTitle: {result['name']}\nURL: {result['url']}\nContent: {result['snippet']}\n" for result in results
]

print("".join(results_prompts))
