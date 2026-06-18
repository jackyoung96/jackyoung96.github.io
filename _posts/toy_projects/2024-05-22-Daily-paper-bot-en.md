---
layout: post
title: "Toy project - Building a Daily paper bot"
tags: archive
lang: en
---

On Huggingface, there is a section called [daily paper](https://huggingface.co/papers).

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/c1491a8c-c0fa-4907-81e5-028ff7c81693)

This is what that section looks like. Every day it curates and posts papers submitted to arXiv or major conferences. It is mostly dominated by papers in natural language processing, multimodal, and vision, but lately some robotics papers show up here and there too. Not long ago, a paper that SKT's multimodal team posted to arXiv even ranked third.

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/b3e2b6fd-aa9f-48ca-9d13-076a6c4264e3)

It is a really nice section, but personally I had three problems with it.
First, you can't check it right away whenever something new is posted (you can subscribe, sure ㅎㅎ but it's by email...).
Second, it's hard to view just the category you want directly.
And finally, the abstract isn't shown immediately — you have to click to read it, and on top of that it's in English so it doesn't read smoothly right away.

So I decided to build a Telegram bot that notifies me right away whenever a paper in a category I want is posted.
The goals are as follows.
1. Periodically parse the Daily paper page and update the DB when new papers are posted
2. Use the abstract to classify papers into five categories: LLM, Multimodal, Computer vision, Reinforcement learning, Robotics
3. Summarize the abstract into 2~3 sentences -> translate into Korean
4. A Telegram bot that, on each paper update, notifies the user about papers in their desired category, in their desired language

Let's get started!!

```text
⛔️ Warning!!
First of all, since I don't come from a developer background, my code may not be perfect (my backend knowledge is very lacking). If you spot any problems in the code, please let me know in the comments!!
```

## Building a Telegram bot

To create a Telegram bot, you need to use the Telegram app's Botfather.
Referring to the content of the source, let's create a bot called daily_paper_bot.
[Source](https://blog.naver.com/PostView.naver?blogId=lifelectronics&logNo=223198582215&redirect=Dlog&widgetTypeCall=true&topReferer=https%3A%2F%2Fwww.google.com%2F&trackingCode=external&directAccess=false)

<img width="457" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/c0a81622-3922-4749-91ff-d8494a672d7a">

Now you can access the bot through the link [t.me/daily_paper_bot](t.me/daily_paper_bot).
And the part I masked out in red is the bot-token of daily_paper_bot. Save this and move on to the next step. (Be careful not to expose it!!)

## Writing the code to store Daily paper user information

To begin with, my grand ambition (?) is that this bot is not just for me but one that anyone can use, so let's write code to receive user information.
In order to send paper information, we need to have:
1) each user's telegram chat id,
2) the paper categories each user wants, and
3) the language each user wants.

Let's create a DB using the freely available sqlite3 and store the user information.

First, we create a table named `telegramchat` in the DB. `DB_NAME` is set as an environment variable.

```python
from dotenv import load_dotenv
load_dotenv()
import os 
import sqlite3
from contextlib import closing

DB_NAME=os.environ.get('DB_NAME')
CATEGORIES = ["LLM", "Multimodal", "Computer vision", "Reinforcement learning", "Robotics"]

with closing(sqlite3.connect(DB_NAME)) as connection:
    with closing(connection.cursor()) as cursor:
        cursor.execute("""CREATE TABLE IF NOT EXISTS telegramchat (
            chatId TEXT NOT NULL,
            lang TEXT DEFAULT EN,
            category TEXT DEFAULT '%s'
            )"""% (','.join(CATEGORIES)))
        connection.commit()
```
For the fields, we'll use the three things mentioned above. For the categories, let's first pick just the 5 categories that frequently show up in daily paper. (Actually, I'm very interested in RL and Robotics, so I snuck them in..ㅎㅎ)

Next, let's write the code that stores the chatID in the DB when a user starts chatting with the bot on telegram,
and the code that lets the user update fields through additional commands.

```python
from dotenv import load_dotenv
import os 
import telegram

import sqlite3
from contextlib import closing

# load .env
load_dotenv()
telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
DB_NAME=os.environ.get('DB_NAME')

# Global variables
CATEGORIES = ["LLM", "Multimodal", "Computer vision", "Reinforcement learning", "Robotics"]
LANGS = ['KO','EN']


async def command_daily_paper(update, context):
    chat_id = update.effective_chat.id
    msg = update.message.text

    # Save chatID
    if msg == '/start':
        with closing(sqlite3.connect(DB_NAME)) as connection:
            with closing(connection.cursor()) as cursor:
                is_exist = cursor.execute("SELECT EXISTS (SELECT 1 FROM telegramchat WHERE chatId = ?)", (chat_id,))

                if not is_exist.fetchone()[0]:
                    cursor.execute(f"INSERT INTO telegramchat (chatId) VALUES (?)", (chat_id,))
                    connection.commit()
        
        bot = telegram.Bot(token = telegram_bot_token)
        message = "Welcome to the daily paper bot!\n\n" + \
                    "Send the category of the papers you are interested in.\n" + \
                    "Possible categories: LLM, Multimodal, Computer vision, Reinforcement learning, Robotics.\n" + \
                    "Send them seperate by comma\n" + \
                    "ex) /setcategory:LLM,Computer vision\n\n" + \
                    "Send the language of the summary you want to get.\n" + \
                    "Possible languages: KO, EN\n" + \
                    "ex) /setlang:KO" 
        await bot.send_message(chat_id, message)

    # Change the Category field
    elif msg.startswith("/setcategory:"):
        categories_str = msg.replace("/setcategory:", "")
        categories = list(set([x.strip() for x in categories_str.split(',')]) & set(CATEGORIES))
        
        if categories:
            # Save data to con
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    cursor.execute("UPDATE telegramchat SET category = ? WHERE chatId = ?", (','.join(categories), chat_id))
                    connection.commit()
            message = f"Category change to {', '.join(categories)}"
        else:
            message = f"Wrong categories input!! Please select categories among LLM, Multimodal, Computer vision, Reinforcement learning, Robotics."  
        bot = telegram.Bot(token = telegram_bot_token)
        await bot.send_message(chat_id, message)

    # Change the Lang field
    elif msg.startswith("/setlang:"):
        lang_str = msg.replace("/setlang:", "").strip()
        
        if lang_str in LANGS:
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    cursor.execute("UPDATE telegramchat SET lang = ? WHERE chatId = ?", (lang_str, chat_id))
                    connection.commit()
            message = f"Language change to {lang_str}"
        else:
            message = f"Wrong language input!! Please select languages among EN, and KO"
        bot = telegram.Bot(token = telegram_bot_token)
        await bot.send_message(chat_id, message)

if __name__ == "__main__":
    application = Application.builder().token(telegram_bot_token).concurrent_updates(True).read_timeout(30).write_timeout(30).build()
    application.add_handler(MessageHandler(filters.Regex("/*") & filters.TEXT, callback=command_daily_paper))
    print("Daily paper telegram bot started!", flush=True)
    application.run_polling()
```

After merging the two code snippets above into one and running it, let's try sending a message to the bot.

<img width="454" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/2dca2537-a010-4463-ba9f-1d91b146282a">

It works well!!


## Writing the Daily paper delivery code

Now let's build the main part, the Daily paper delivery code. The delivery code consists of three parts.
1. Parse the daily paper page and check whether papers have been updated
2. Use the parsed content to classify categories and summarize
3. Send to users

Let's go through them one by one, step by step.

### Parsing Daily paper

First, let's scrape all of today's daily papers.

```python
from datetime import datetime, timedelta
import requests

def fetch_data():
    
    fetch_day = datetime.now()
    
    result = None
    for _ in range(7): 
        day_str = fetch_day.strftime("%Y-%m-%d")
        url = f"https://huggingface.co/papers?date={day_str}"
        response = requests.get(url)
        if response.status_code == 200:
            result = response.text
            break
        
        fetch_day -= timedelta(days=1)
        
    return fetch_day, result
```

Sometimes the date doesn't get updated, so just in case, the code is set up to check dates as far back as 7 days. Since the text fetched this way is raw html, we additionally need code to parse the paper title, paper link, and abstract out of the html. We can parse those contents using BeautifulSoup.

```python
from bs4 import BeautifulSoup

# Fetch the paper abstract
def fetch_paper_abstract(paper_url):
    response = requests.get(paper_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        abstract_tag = soup.find('p', class_='text-gray-700 dark:text-gray-400')
        if abstract_tag:
            abstract_tag = abstract_tag.get_text(strip=True)
            abstract_tag = abstract_tag.replace('\n', " ")
            return abstract_tag
    return "Abstract not found."

# Parse papers from the main page
def parse_papers(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    papers = []
    articles = soup.find_all('article', class_='flex flex-col overflow-hidden rounded-xl border')
    
    for article in articles:
        paper_name_tag = article.find('h3')
        if paper_name_tag:
            paper_name = paper_name_tag.get_text(strip=True)
            paper_url_tag = paper_name_tag.find('a')
            if paper_url_tag and paper_url_tag.has_attr('href'):
                paper_url = "https://huggingface.co" + paper_url_tag['href']
                paper_abstract = fetch_paper_abstract(paper_url)
                papers.append((paper_name, paper_url, paper_abstract))
    return papers
```

Now we need to check whether each of these contents existed before or not. If we don't do this, we'd keep parsing, summarizing, and translating papers that were posted before, and keep paying OpenAI costs for them. The OpenAI API really leaks money like water if you just use it carelessly...

Just like when receiving user information, let's create a new table named dailypaper in the DB. The fields are paper title, paper update date, English summary, Korean summary, and category.

```python
from dotenv import load_dotenv
load_dotenv()
import os 
import sqlite3
from contextlib import closing

DB_NAME=os.environ.get('DB_NAME')

with closing(sqlite3.connect(DB_NAME)) as connection:
    with closing(connection.cursor()) as cursor:
        cursor.execute("""CREATE TABLE IF NOT EXISTS dailypaper (
            title TEXT NOT NULL,
            date DATE,
            summaryEN TEXT,
            summaryKO TEXT,
            categories TEXT
            )""")
        connection.commit()

def update_paper():
    
    fetch_day, url_content = fetch_data()
    
    new_papers = []
    if url_content:
        papers = parse_papers(url_content)
        
        for paper_name, paper_url, paper_abstract in papers:
            
            # TODO: check if the paper is in the DB -> if so, pass
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    is_exist = cursor.execute("SELECT EXISTS (SELECT 1 FROM dailypaper WHERE title = ?)", (paper_name,))
                    is_exist = is_exist.fetchone()[0]
            if is_exist:
                continue
            
            # Summarization and category classification code
            summary = summarize_text(paper_abstract)
            translate_summary = translate_text(summary)
            categories = categorize_paper(title=paper_name, summary=summary)
            categories_str = ','.join(categories)
            
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    cursor.execute(f"INSERT INTO dailypaper VALUES (?,?,?,?,?)", (paper_name, fetch_day.strftime("%Y-%m-%d"), summary, translate_summary, categories_str))
                    connection.commit()
                    
            new_papers.append({
                "title": paper_name,
                "summary_EN": summary,
                "summary_KO": translate_summary,
                "categories": categories,
                "url": paper_url
            })
    
    return new_papers
```

Now, once we have the summarization, translation, and category classification code, we'll be able to update these contents.

### Daily paper summarization, translation, and category classification

For summarization and translation, let's use the OpenAI API. `OPENAI_API_KEY` is also managed as an environment variable. (**If this leaks anywhere, it's really a huge problem!!**)

```python
import openai

openai.api_key = os.environ.get('OPENAI_API_KEY')

def summarize_text(text):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", 
             "content": "You are a highly knowledgeable assistant who is very specialized in deep learning field. "
                        "Provide the summarization of the given content into 2~3 sentences. "
                        "ONLY provide the summmarized sentences."},
            {"role": "user", "content": f"Summarize this content into maximum 2 sentences: {text}"},
        ]
    )
    # summary = response.choices[0].message.content.strip()
    summary = text.split('.')[0]
    
    return summary

def translate_text(text):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", 
             "content": "You are a highly knowledgeable assistant who is very specialized in English-Korean translating. "
                        "Provide translated text of the given content. "
                        "Don't translate English terminologies and focus on translating common words. "
                        "ONLY provide translated sentences"},
            {"role": "user", "content": f"Translate it into Korean: {text}"},
        ]
    )
    
    translated_text = response.choices[0].message.content.strip()
    
    return translated_text

def categorize_paper(title, summary):
    messages=[
        {"role": "system", 
         "content": "You are a highly knowledgeable assistant who is very specialized in deep learning field. "
                    "Suggest one or multiple categories of the given paper. "
                    f"Categories must be selected among {str(CATEGORIES)}. "
                    "ONLY provide categories seperated by comma and nothing else."},
        {"role": "user", 
         "content": "What categories would you suggest me to add to this paper?\n"
                    f"paper title: {title}\n"
                    f"paper summary: {summary}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    categories = response.choices[0].message.content.strip()
    categories = categories.split(",")
    categories = [c.strip() for c in categories]
    
    return categories
```

When using the OpenAI API, you can now use it while comparing model performance and price.
In my case, after trying it out, gpt-3.5-turbo was enough for category classification, while using gpt-4o for summarization and translation was more satisfying in terms of quality. As for price, gpt-4o is 10 times more expensive...

I didn't do separate engineering on the prompts for summarization, translation, and category classification. Of course there are more efficient prompts out there, but these days the models understand and handle things just fine even if you explain things roughly.

### Sending via the Daily paper telegram bot

Now all that's left is to send the updated content.

```python
import telegram
import asyncio

async def send_daily_message(user_info, new_papers):
    chat_id, lang, categories_str = user_info
    categories = categories_str.strip().split(',')
    
    token = telegram_bot_token
    bot = telegram.Bot(token = token)
    
    for new_paper in new_papers:
        # Whether it's included in the category (if None, treat as all categories)
        paper_categories = new_paper.get('categories')
        if len(set(paper_categories) & set(categories)) == 0:
            continue
        
        paper_name = new_paper.get('title')
        summary = new_paper.get(f'summary_{lang}')
        paper_url = new_paper.get('url')

        message = f"**{paper_name}**\n{summary}\n\n{paper_url}"
        await bot.send_message(chat_id, message, parse_mode='Markdown')

async def main():
    while True:
        print("Checking daily paper update...", flush=True)
        new_paper = update_paper()
        
        with closing(sqlite3.connect(DB_NAME)) as connection:
            with closing(connection.cursor()) as cursor:
                res = cursor.execute("SELECT chatId, lang, category FROM telegramchat")
                all_user_info = res.fetchall()
        
        if new_paper:
            print("Get new papers!!", flush=True)
                    
            for user_info in all_user_info:
                await send_daily_message(user_info, new_paper)
        
        else:
            print("There is nothing new...", flush=True)

        await asyncio.sleep(10 * 60) # wait 10 minutes
        
if __name__ == "__main__":
    asyncio.run(main())
```

Once every 10 minutes, I parsed the daily paper page, checked whether there was any updated content, and then performed summarization, translation, and category classification.
After that, for papers that fall into the category each user wants, I implemented it to send the paper's title, summary, and link.

<img width="486" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/00e61f4f-602e-4fea-b3f9-7db37c02f473">

You can see that messages arrive nicely in this way.

## Full code

The entire code consists of the bot code `dailypaper_userbot.py` that receives user information, and the bot code `dailypaper_sendbot.py` that delivers dailypaper information. You just run each of them.

```python
### dailypaper_userbot.py

from dotenv import load_dotenv
import os 
from collections import defaultdict

import telegram
from telegram.ext import (
    Application, 
    MessageHandler, 
    filters,
)

import sqlite3
from contextlib import closing


# load .env
load_dotenv()
telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
DB_NAME=os.environ.get('DB_NAME')

# Global variables
CATEGORIES = ["LLM", "Multimodal", "Computer vision", "Reinforcement learning", "Robotics"]
LANGS = ['KO','EN']

with closing(sqlite3.connect(DB_NAME)) as connection:
    with closing(connection.cursor()) as cursor:
        cursor.execute("""CREATE TABLE IF NOT EXISTS telegramchat (
            chatId TEXT NOT NULL,
            lang TEXT DEFAULT EN,
            category TEXT DEFAULT '%s'
            )"""% (','.join(CATEGORIES)))
        connection.commit()

async def command_daily_paper(update, context):
    chat_id = update.effective_chat.id
    msg = update.message.text

    if msg == '/start':
        # Save data to con
        with closing(sqlite3.connect(DB_NAME)) as connection:
            with closing(connection.cursor()) as cursor:
                is_exist = cursor.execute("SELECT EXISTS (SELECT 1 FROM telegramchat WHERE chatId = ?)", (chat_id,))

                if not is_exist.fetchone()[0]:
                    cursor.execute(f"INSERT INTO telegramchat (chatId) VALUES (?)", (chat_id,))
                    connection.commit()
        
        bot = telegram.Bot(token = telegram_bot_token)
        message = "Welcome to the daily paper bot!\n\n" + \
                    "Send the category of the papers you are interested in.\n" + \
                    "Possible categories: LLM, Multimodal, Computer vision, Reinforcement learning, Robotics.\n" + \
                    "Send them seperate by comma\n" + \
                    "ex) /setcategory:LLM,Computer vision\n\n" + \
                    "Send the language of the summary you want to get.\n" + \
                    "Possible languages: KO, EN\n" + \
                    "ex) /setlang:KO" 
        await bot.send_message(chat_id, message)
    elif msg.startswith("/setcategory:"):
        categories_str = msg.replace("/setcategory:", "")
        categories = list(set([x.strip() for x in categories_str.split(',')]) & set(CATEGORIES))
        
        if categories:
            # Save data to con
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    cursor.execute("UPDATE telegramchat SET category = ? WHERE chatId = ?", (','.join(categories), chat_id))
                    connection.commit()
            message = f"Category change to {', '.join(categories)}"
        else:
            message = f"Wrong categories input!! Please select categories among LLM, Multimodal, Computer vision, Reinforcement learning, Robotics."  
        bot = telegram.Bot(token = telegram_bot_token)
        await bot.send_message(chat_id, message)
    elif msg.startswith("/setlang:"):
        lang_str = msg.replace("/setlang:", "").strip()
        
        if lang_str in LANGS:
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    cursor.execute("UPDATE telegramchat SET lang = ? WHERE chatId = ?", (lang_str, chat_id))
                    connection.commit()
            message = f"Language change to {lang_str}"
        else:
            message = f"Wrong language input!! Please select languages among EN, and KO"
        bot = telegram.Bot(token = telegram_bot_token)
        await bot.send_message(chat_id, message)

if __name__ == "__main__":
    application = Application.builder().token(telegram_bot_token).concurrent_updates(True).read_timeout(30).write_timeout(30).build()
    application.add_handler(MessageHandler(filters.Regex("/*") & filters.TEXT, callback=command_daily_paper))
    print("Daily paper telegram bot started!", flush=True)
    application.run_polling()
```

```python
### dailypaper_sendbot.py

from dotenv import load_dotenv
import os 
import openai

import telegram
import asyncio

import sqlite3
from contextlib import closing

from bs4 import BeautifulSoup
from datetime import datetime, timedelta
import requests

# load .env
load_dotenv()
openai.api_key = os.environ.get('OPENAI_API_KEY')
telegram_bot_token = os.environ.get('TELEGRAM_BOT_TOKEN')
DB_NAME=os.environ.get('DB_NAME')

CATEGORIES = ["LLM", "Multimodal", "Computer vision", "Reinforcement learning", "Robotics"]

with closing(sqlite3.connect(DB_NAME)) as connection:
    with closing(connection.cursor()) as cursor:
        cursor.execute("""CREATE TABLE IF NOT EXISTS dailypaper (
            title TEXT NOT NULL,
            date DATE,
            summaryEN TEXT,
            summaryKO TEXT,
            categories TEXT
            )""")
        connection.commit()

def fetch_data():
    
    fetch_day = datetime.now()
    
    result = None
    for _ in range(7): 
        day_str = fetch_day.strftime("%Y-%m-%d")
        url = f"https://huggingface.co/papers?date={day_str}"
        response = requests.get(url)
        if response.status_code == 200:
            result = response.text
            break
        
        fetch_day -= timedelta(days=1)
        
    return fetch_day, result

# Fetch the paper abstract
def fetch_paper_abstract(paper_url):
    response = requests.get(paper_url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        abstract_tag = soup.find('p', class_='text-gray-700 dark:text-gray-400')
        if abstract_tag:
            abstract_tag = abstract_tag.get_text(strip=True)
            abstract_tag = abstract_tag.replace('\n', " ")
            return abstract_tag
    return "Abstract not found."

# Parse papers from the main page
def parse_papers(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    papers = []
    articles = soup.find_all('article', class_='flex flex-col overflow-hidden rounded-xl border')
    
    for article in articles:
        paper_name_tag = article.find('h3')
        if paper_name_tag:
            paper_name = paper_name_tag.get_text(strip=True)
            paper_url_tag = paper_name_tag.find('a')
            if paper_url_tag and paper_url_tag.has_attr('href'):
                paper_url = "https://huggingface.co" + paper_url_tag['href']
                paper_abstract = fetch_paper_abstract(paper_url)
                papers.append((paper_name, paper_url, paper_abstract))
    return papers

def summarize_text(text):
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {"role": "system", 
             "content": "You are a highly knowledgeable assistant who is very specialized in deep learning field. "
                        "Provide the summarization of the given content into 2~3 sentences. "
                        "ONLY provide the summmarized sentences."},
            {"role": "user", "content": f"Summarize this content into maximum 2 sentences: {text}"},
        ]
    )
    # summary = response.choices[0].message.content.strip()
    summary = text.split('.')[0]
    
    return summary

def translate_text(text):
    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {"role": "system", 
             "content": "You are a highly knowledgeable assistant who is very specialized in English-Korean translating. "
                        "Provide translated text of the given content. "
                        "Don't translate English terminologies and focus on translating common words. "
                        "ONLY provide translated sentences"},
            {"role": "user", "content": f"Translate it into Korean: {text}"},
        ]
    )
    
    translated_text = response.choices[0].message.content.strip()
    
    return translated_text

def categorize_paper(title, summary):
    messages=[
        {"role": "system", 
         "content": "You are a highly knowledgeable assistant who is very specialized in deep learning field. "
                    "Suggest one or multiple categories of the given paper. "
                    f"Categories must be selected among {str(CATEGORIES)}. "
                    "ONLY provide categories seperated by comma and nothing else."},
        {"role": "user", 
         "content": "What categories would you suggest me to add to this paper?\n"
                    f"paper title: {title}\n"
                    f"paper summary: {summary}"}
    ]
    response = openai.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=messages
    )
    categories = response.choices[0].message.content.strip()
    categories = categories.split(",")
    categories = [c.strip() for c in categories]
    
    return categories

def update_paper():
    
    fetch_day, url_content = fetch_data()
    
    new_papers = []
    if url_content:
        papers = parse_papers(url_content)
        
        for paper_name, paper_url, paper_abstract in papers:
            
            # TODO: check if the paper is in the DB -> if so, pass
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    is_exist = cursor.execute("SELECT EXISTS (SELECT 1 FROM dailypaper WHERE title = ?)", (paper_name,))
                    is_exist = is_exist.fetchone()[0]
            if is_exist:
                continue
            
            summary = summarize_text(paper_abstract)
            translate_summary = translate_text(summary)
            categories = categorize_paper(title=paper_name, summary=summary)
            categories_str = ','.join(categories)
            
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    cursor.execute(f"INSERT INTO dailypaper VALUES (?,?,?,?,?)", (paper_name, fetch_day.strftime("%Y-%m-%d"), summary, translate_summary, categories_str))
                    connection.commit()
                    
            new_papers.append({
                "title": paper_name,
                "summary_EN": summary,
                "summary_KO": translate_summary,
                "categories": categories,
                "url": paper_url
            })
    
    return new_papers

async def send_daily_message(user_info, new_papers):
    chat_id, lang, categories_str = user_info
    categories = categories_str.strip().split(',')
    
    token = telegram_bot_token
    bot = telegram.Bot(token = token)
    
    for new_paper in new_papers:
        # Whether it's included in the category (if None, treat as all categories)
        paper_categories = new_paper.get('categories')
        if len(set(paper_categories) & set(categories)) == 0:
            continue
        
        paper_name = new_paper.get('title')
        summary = new_paper.get(f'summary_{lang}')
        paper_url = new_paper.get('url')

        message = f"**{paper_name}**\n{summary}\n\n{paper_url}"
        await bot.send_message(chat_id, message, parse_mode='Markdown')

async def main():
    while True:
        print("Checking daily paper update...", flush=True)
        new_paper = update_paper()
        
        with closing(sqlite3.connect(DB_NAME)) as connection:
            with closing(connection.cursor()) as cursor:
                res = cursor.execute("SELECT chatId, lang, category FROM telegramchat")
                all_user_info = res.fetchall()
        
        if new_paper:
            print("Get new papers!!", flush=True)
                    
            for user_info in all_user_info:
                await send_daily_message(user_info, new_paper)
        
        else:
            print("There is nothing new...", flush=True)

        await asyncio.sleep(10 * 60) # wait 10 minutes
        
if __name__ == "__main__":
    asyncio.run(main())
```


# Conclusion

This is how I built a Telegram bot. In the case of sqlite3, since it works synchronously, there seem to be parts where it doesn't mesh well with the asynchronously-working telegram bot. This is an area that needs improvement going forward.

Also, for now I just left it running locally, but if I host it on a single AWS EC2 instance, lots of people could use it.
