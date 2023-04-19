import csv

import emoji as emoji
import pandas as pd
import json
import praw
import requests
import emoji
import re

if __name__ == '__main__':

    app_id = 'Yymx6smwc98BSpn8lTWgFg'
    secret = 'gN8-ysrh8bJKcuk9j95HAnP7DFkPXg'
    auth = requests.auth.HTTPBasicAuth(app_id, secret)
    reddit_username = 'SimonEducational'
    reddit_password = 'PurdueNLP123!'
    data = {
    'grant_type': 'password',
    'username': reddit_username,
    'password': reddit_password
    }
    headers = {'User-Agent': 'EmojiProject/0.0.1'}
    res = requests.post('https://www.reddit.com/api/v1/access_token',
    auth=auth, data=data, headers=headers)
    TOKEN = res.json()['access_token']
    print(res)
    res.json()
    headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}
    res = requests.get('https://oauth.reddit.com/api/v1/me', headers=headers)
    df = pd.DataFrame()  # initialize dataframe
    all_comments = pd.DataFrame()

    reddit = praw.Reddit(
        client_id="Yymx6smwc98BSpn8lTWgFg",
        client_secret="gN8-ysrh8bJKcuk9j95HAnP7DFkPXg",
        password="PurdueNLP123!",
        user_agent="Comment Extraction (by u/USERNAME)",
        username="SimonEducational",
    )

    # loop through each post retrieved from GET request

    params = {'limit': 100}



    j = 0

    d = {}
    c = {}

    replacements = [
        ('\n', ' '),
        (',', ''),
        ('::', ': :'),
    ]

    bad_words = [
        'cock', 'dick', 'cum', 'slut', 'fuck', 'balls'
    ]

    for i in range(10):
        res1 = requests.get("https://oauth.reddit.com/r/emojipasta/new",
                            headers=headers, params=params)
        for post in res1.json()['data']['children']:
            print(j)
            url = f"https://www.reddit.com{post['data']['permalink']}"
            submission = reddit.submission(url=url)
            submission.comments.replace_more(limit=None)
            for comment in submission.comments.list():
                comm = emoji.demojize(re.sub(':', '', comment.body)).encode('ascii', 'ignore').decode('ascii').strip().lower()
                link = emoji.demojize(re.sub(':', '', post['data']['permalink'])).encode('ascii', 'ignore').decode('ascii').strip()
                if 'chainlink' or 'I am a bot' in comm:
                    continue
                if len(comm) < 20:
                    continue
                for old, new in replacements:
                    comm = re.sub(old, new, comm)
                    link = re.sub(old, new, link)
                colon_count = 0
                length = len(comm)
                elements = 0
                while elements < length:
                    #print(f'elements = {elements}, length = {length}, colon count = {colon_count}, element = {comm[elements]}')
                    if comm[elements] == ':':
                        colon_count += 1
                        if colon_count % 2 == 1:
                            comm = comm[:(elements)] + ' ' + comm[(elements):]
                            elements += 1
                            length += 1
                        else:
                            comm = comm[:(elements + 1)] + ' ' + comm[(elements + 1):]
                            elements += 1
                            length += 1
                    elements += 1
                comm_list = comm.split()
                emoji_count = 0
                profanity_count = 0
                for word in comm_list:
                    if ':' in word:
                        emoji_count += 1
                    else:
                        for bad_word in bad_words:
                            if bad_word in word:
                                print(word)
                                profanity_count += 1
                if profanity_count >= 5:
                    continue
                emoji_ratio = emoji_count / len(comm_list)
                if emoji_ratio < 0.1 or emoji_ratio > 0.6:
                    continue
                for bad_word in bad_words:
                    comm = re.sub(bad_word, '', comm)
                c[j] = {'comment': comm,
                        'permalink': link}
            # append relevant data to dataframe
            sub = emoji.demojize(re.sub(':', '', post['data']['subreddit'])).encode('ascii', 'ignore').decode('ascii').strip()
            title = emoji.demojize(re.sub(':', '', post['data']['title'])).encode('ascii', 'ignore').decode('ascii').strip()
            selftext = emoji.demojize(re.sub(':', '', post['data']['selftext'])).encode('ascii', 'ignore').decode('ascii').strip().lower()
            permalink = emoji.demojize(re.sub(':', '', post['data']['permalink'])).encode('ascii', 'ignore').decode('ascii').strip()
            if 'chainlink' in selftext:
                print(f'Chainlink {selftext}')
                j = j + 1
                continue
            if 'request' in title:
                print(f'request {title}')
                j = j + 1
                continue
            for old, new in replacements:
                sub = re.sub(old, new, sub)
                title = re.sub(old, new, title)
                selftext = re.sub(old, new, selftext)
                permalink = re.sub(old, new, permalink)
            colon_count = 0
            length = len(selftext)
            elements = 0
            while elements < length:
                #print(
                #    f'elements = {elements}, length = {length}, colon count = {colon_count}, element = {selftext[elements]}')
                if selftext[elements] == ':':
                    colon_count += 1
                    if colon_count % 2 == 1:
                        selftext = selftext[:(elements)] + ' ' + selftext[(elements):]
                        elements += 1
                        length += 1
                    else:
                        selftext = selftext[:(elements + 1)] + ' ' + selftext[(elements + 1):]
                        elements += 1
                        length += 1
                elements += 1
            text_list = selftext.split()
            emoji_count = 0
            profanity_count = 0
            for word in text_list:
                if ':' in word:
                    emoji_count += 1
                else:
                    for bad_word in bad_words:
                        if bad_word in word:
                            print(word)
                            profanity_count += 1
            if profanity_count >= 5:
                print(f'Profanity Count >= 5 {selftext}, Prof count = {profanity_count}')
                j = j + 1
                continue
            if len(text_list) == 0:
                j = j + 1
                continue
            emoji_ratio = emoji_count / len(text_list)
            if emoji_ratio < 0.1 or emoji_ratio > 0.6:
                print(f'Emoji Ratio is out of bounds {selftext} emoji ratio = {emoji_ratio}')
                j = j + 1
                continue
            for bad_word in bad_words:
                selftext = re.sub(bad_word, '', selftext)
            d[j] = {
                'subreddit': sub,
                'title': title,
                'selftext': selftext,
                'permalink': permalink}
            j = j + 1
            fullname = post['kind'] + '_' + post['data']['id']
            # add/update fullname in params
            params['after'] = fullname


    df = pd.DataFrame.from_dict(d, orient='index', columns=['subreddit', 'title', 'selftext', 'permalink'])
    all_comments = pd.DataFrame.from_dict(c, orient='index', columns=['comment', 'permalink'])

    df.to_csv('Posts.csv')
    all_comments.to_csv('Comments.csv')

    print(df.shape)
