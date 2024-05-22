---
layout: post
title: Toy project - Daily paper bot 만들기
tags: archive
---

Huggingface 에 보면 [daily paper](https://huggingface.co/papers) 라는 코너가 있습니다.

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/c1491a8c-c0fa-4907-81e5-028ff7c81693)

이런식으로 된 코너인데, 매일매일 아카이브나 유명 학회에 제출된 논문들을 추려서 올려줍니다. 특히 자연어 처리 분야나 멀티모달, 비전 쪽의 논문이 주를 이루는데, 요즘에는 로보틱스 논문도 간간히 올라옵니다. 얼마전에는 SKT 멀티모달 팀에서 아카이브에 올린 논문이 3위에 랭크되기도 했었습니다.

![image](https://github.com/snulion-study/algorithm-adv/assets/57203764/b3e2b6fd-aa9f-48ca-9d13-076a6c4264e3)

굉장히 좋은 포스팅이기는 한데 개인적으로는 세 가지 문제가 있었습니다.  
첫 번째로는 새로운게 올라올 때마다 바로바로 확인이 안된다는 것 (구독하면 되긴 합니다 ㅎㅎ 메일이라 그렇지...)  
두 번째로는 원하는 카테고리만 바로 보기 어렵다는 것  
마지막으로 abstract 이 바로 보이지 않고 클릭해서 읽어야 하는데다가 영어라 바로바로 읽히지는 않는다는 것  

그래서 원하는 카테고리 논문이 올라오면 바로 알려주는 텔레그램 봇을 한 번 만들어 보기로 했습니다.
목표는 아래와 같습니다.
1. Daily paper 페이지를 주기적으로 parsing 해서 새로운 논문이 올라오면 DB 에 업데이트
2. Abstract 를 이용해 LLM, Multimodal, Computer vision, Reinforcement learning, Robotics 다섯가지 카테고리로 paper 를 분류
3. Abstract 를 2~3 문장으로 요약 -> 한글로 번역
4. 논문 업데이트마다 사용자가 원하는 카테고리의 논문 요약을, 원하는 언어로 알려주는 텔레그램 봇

한 번 시작해 보겠습니다!!

```text
⛔️ 주의!!
일단 저는 개발자 출신이 아니다보니 코드가 완벽하지 않을 수 있습니다 (백엔드 지식이 매우 부족합니다) 혹시 코드에 문제가 보이시면 댓글로 알려주세요!!
```

## Telegram bot 만들기

Telegram bot 을 만들기 위해서는 Telegram 앱의 Botfather 를 이용해야 합니다.
출처의 내용을 참고해서 daily_paper_bot 이라는 봇을 만들어보도록 하겠습니다. 
[출처](https://blog.naver.com/PostView.naver?blogId=lifelectronics&logNo=223198582215&redirect=Dlog&widgetTypeCall=true&topReferer=https%3A%2F%2Fwww.google.com%2F&trackingCode=external&directAccess=false)

<img width="457" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/c0a81622-3922-4749-91ff-d8494a672d7a">

자 이제 [t.me/daily_paper_bot](t.me/daily_paper_bot) 이라는 링크를 통해 봇에 접속할 수 있습니다.  
그리고 빨간색으로 가려놓은 부분이 daily_paper_bot 의 bot-token 입니다. 이걸 저장해두고 다음 스텝으로 넘어갑니다. (노출되지 않도록 주의!!)

## Daily paper user 정보 저장 코드 만들기

일단 원대한 포부(?)는 저만 사용하는 봇이 아니라 누구나 사용할 수 있는 봇이기 때문에, 유저 정보를 받는 코드를 만들도록 하겠습니다.
Paper 정보를 보내기 위해서는 
1) 각 유저의 telegram chat id, 
2) 각 유저가 원하는 paper 카테고리, 
3) 각 유저가 원하는 언어 정보
   
를 가지고 있어야 합니다. 무료로 사용 가능한 sqlite3 를 사용해 DB 를 만들고 유저 정보를 저장해보도록 하겠습니다.

우선 DB 에 `telegramchat` 이라는 이름의 table 을 만들어줍니다. `DB_NAME` 은 환경변수로 설정합니다. 

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
Field 로는 위에서 말한 3가지를 사용하겠습니다. 카테고리의 경우 daily paper 에 자주 올라오는 5개 카테고리만 우선 선정하겠습니다. (사실 제가 RL 이랑 Robotics 관심이 많아서 끼워넣었습니다..ㅎㅎ)

다음은 telegram 에서 유저가 봇과 채팅을 시작했을 때 DB 에 chatID 가 저장되는 코드와
유저의 추가적인 명령어를 통해 필드를 업데이트 할 수 있는 코드를 만들겠습니다.

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

    # chatID 저장
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

    # Category 필드 변경
    elif msg.startswith("/setcategory:"):
        categories_str = msg.replace("/setcategory:", "")
        categories = list(set([x.strip() for x in categories_str.split(',')]) & set(CATEGORIES))
        
        if categories:
            # con 에 데이터 저장
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    cursor.execute("UPDATE telegramchat SET category = ? WHERE chatId = ?", (','.join(categories), chat_id))
                    connection.commit()
            message = f"Category change to {', '.join(categories)}"
        else:
            message = f"Wrong categories input!! Please select categories among LLM, Multimodal, Computer vision, Reinforcement learning, Robotics."  
        bot = telegram.Bot(token = telegram_bot_token)
        await bot.send_message(chat_id, message)

    # Lang 필드 변경
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

위의 두 코드를 하나로 합쳐서 실행시킨 후 한 번 봇에게 메시지를 보내보겠습니다.

<img width="454" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/2dca2537-a010-4463-ba9f-1d91b146282a">

잘 작동하는군요!!


## Daily paper 전송 코드 만들기

이제 메인 파트인 Daily paper 전송 코드를 만들어 보겠습니다. 전송 코드는 3가지 파트로 이루어집니다.
1. daily paper 페이지를 parsing 하고 논문 업데이트 여부를 확인
2. parsing 한 내용을 가지고 카테고리 분류 및 요약 작업
3. 유저들에게 전송

하나하나 차근차근 해보도록 하겠습니다.

### Daily paper 파싱

우선 오늘 날짜의 daily paper 를 전부 긁어오도록 하겠습니다. 

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

가끔 날짜가 업데이트가 안되는 경우들이 있더라구요, 혹시 몰라서 7일 전 날짜까지 확인해보도록 코드가 구성되어 있습니다. 이렇게 가져온 텍스트는 날것의 html 이기 때문에 paper title과 paper link, abstract을 html 에서 파싱하는 코드가 추가적으로 필요합니다. BeautifulSoup 를 사용해서 해당 내용들을 파싱하면 됩니다.

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

이제 해당 내용들이 이전에 있었는지 없었는지 확인하는 작업이 필요합니다. 이걸 안하면 이전에 올라왔던 논문들도 계속 파싱하면서 요약하고 번역하는데 OpenAI 비용을 계속 지불하게 되겠죠. OpenAI API 는 진짜 대충 쓰다보면 물새듯이 돈이 나갑니다...

User 정보를 받을 때와 마찬가지로 DB 에 dailypaper 라는 이름의 table 을 새로 하나 만들어주겠습니다. Field 는 논문 제목, 논문 업데이트 일자, 영문 요약, 한글 요약, 카테고리 입니다.

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
            
            # TODO: paper DB에 있는지 확인 -> 있으면 pass
            with closing(sqlite3.connect(DB_NAME)) as connection:
                with closing(connection.cursor()) as cursor:
                    is_exist = cursor.execute("SELECT EXISTS (SELECT 1 FROM dailypaper WHERE title = ?)", (paper_name,))
                    is_exist = is_exist.fetchone()[0]
            if is_exist:
                continue
            
            # 요약, 카테고리 구분 코드
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

이제 요약, 번역, 카테고리 분류 코드만 있으면 해당 내용을 업데이트 할 수 있게 되었습니다.

### Daily paper 요약, 번역, 카테고리 분류

요약과 번역 작업은 openAI API 를 사용하도록 하겠습니다. `OPENAI_API_KEY` 도 환경변수로 관리합니다. (**이거는 어디에 새어나가면 정말 큰일납니다!!**)

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

OpenAI API 를 사용할 때에는 이제 모델의 성능과 가격을 비교하면서 사용하시면 됩니다.
저같은 경우는 해보니까 카테고리 분류는 gpt-3.5-turbo 로도 충분했고, 요약과 번역은 gpt-4o 를 사용하는게 품질면에서 더 만족스러웠습니다. 가격은 gpt-4o 가 10 배 비쌉니다...

요약, 번역, 카테고리 분류에 대한 프롬프트는 따로 엔지니어링을 진행하지는 않았습니다. 물론 더 효율적인 프롬프트가 있겠지만 요새 모델들은 대충 말해도 척척 알아듣습니다.

### Daily paper telegram bot 전송

자 이제 업데이트 된 내용을 전송만 해주면 됩니다.

```python
import telegram
import asyncio

async def send_daily_message(user_info, new_papers):
    chat_id, lang, categories_str = user_info
    categories = categories_str.strip().split(',')
    
    token = telegram_bot_token
    bot = telegram.Bot(token = token)
    
    for new_paper in new_papers:
        # category 에 포함되는지 (None 인 경우 모든 category 로 간주)
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

        await asyncio.sleep(10 * 60) # 10분 대기
        
if __name__ == "__main__":
    asyncio.run(main())
```

10분마다 한 번씩 daily paper 페이지를 파싱하고, 업데이트 된 내용이 있는지 확인한 후 요약과 번역, 카테고리 분류를 수행했습니다. 
이후 각 유저들이 원하는 카테고리에 해당되는 논문일 경우 논문의 제목과 요약, 링크를 보내주도록 구현했습니다.

<img width="486" alt="image" src="https://github.com/snulion-study/algorithm-adv/assets/57203764/00e61f4f-602e-4fea-b3f9-7db37c02f473">

이런식으로 메시지가 잘 오는 걸 확인할 수 있습니다.

## Full code

전체 코드는 user 정보를 받는 bot 코드 `dailypaper_userbot.py` 와 dailypaper 정보 전달 bot 코드 `dailypaper_sendbot.py` 로 이루어집니다. 각각을 실행시켜주면 됩니다.

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
        # con 에 데이터 저장
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
            # con 에 데이터 저장
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
            
            # TODO: paper DB에 있는지 확인 -> 있으면 pass
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
        # category 에 포함되는지 (None 인 경우 모든 category 로 간주)
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

        await asyncio.sleep(10 * 60) # 10분 대기
        
if __name__ == "__main__":
    asyncio.run(main())
```


# Conclusion

이렇게 텔레그램 봇을 하나 만들어보았습니다. sqlite3 의 경우 동기 방식으로 작동하기 때문에 비동기로 작동하는 telegram bot 와 맞지 않는 부분이 있는 것 같습니다. 앞으로 개선이 필요한 부분입니다.

또한 지금은 그냥 로컬에서 실행시켜뒀지만, AWS 의 EC2 하나에 띄워두면 많은 분들이 사용할 수 있겠습니다. 