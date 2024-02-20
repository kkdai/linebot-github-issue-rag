# -*- coding: utf-8 -*-

#  Licensed under the Apache License, Version 2.0 (the "License"); you may
#  not use this file except in compliance with the License. You may obtain
#  a copy of the License at
#
#       https://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS, WITHOUT
#  WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied. See the
#  License for the specific language governing permissions and limitations
#  under the License.

from langchain.callbacks import StdOutCallbackHandler
from langchain.chains import RetrievalQA
from langchain.llms.openai import OpenAIChat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.runnables import RunnableLambda, RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_community.vectorstores import FAISS
from operator import itemgetter
from dotenv import load_dotenv, find_dotenv
from linebot.models import (
    MessageEvent, TextMessage, TextSendMessage,
)
from linebot.exceptions import (
    InvalidSignatureError
)
from linebot.aiohttp_async_http_client import AiohttpAsyncHttpClient
from linebot import (
    AsyncLineBotApi, WebhookParser
)
import os
import sys

import aiohttp

from fastapi import Request, FastAPI, HTTPException
from langchain_community.document_loaders import GitHubIssuesLoader

# Get environment "GITHUB_TOKEN"
github_token = os.getenv("GITHUB_TOKEN")

loader = GitHubIssuesLoader(
    repo="kkdai/bookmarks",
    # delete/comment out this argument if you've set the access token as an env var.
    access_token=github_token,
    creator="kkdai",
    include_prs=False,
)

docs = loader.load()
print("Prepare Github Issue Total Num:" + len(docs))

# get channel_secret and channel_access_token from your environment variable
channel_secret = os.getenv('ChannelSecret', None)
channel_access_token = os.getenv('ChannelAccessToken', None)
if channel_secret is None:
    print('Specify LINE_CHANNEL_SECRET as environment variable.')
    sys.exit(1)
if channel_access_token is None:
    print('Specify LINE_CHANNEL_ACCESS_TOKEN as environment variable.')
    sys.exit(1)

app = FastAPI()
session = aiohttp.ClientSession()
async_http_client = AiohttpAsyncHttpClient(session)
line_bot_api = AsyncLineBotApi(channel_access_token, async_http_client)
parser = WebhookParser(channel_secret)

# Langchain (you must use 0613 model to use OpenAI functions.)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50,
    length_function=len
)

ecicoder_chunks = text_splitter.transform_documents(docs)
# print first issue for test
print(ecicoder_chunks[0])


embeddings = OpenAIEmbeddings()
vectorstore = FAISS.from_documents(docs, embeddings)
vectorstore.add_documents(ecicoder_chunks)

retriever = vectorstore.as_retriever()
handler = StdOutCallbackHandler()
llm = ChatOpenAI()

qa_with_sources_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    callbacks=[handler],
    return_source_documents=True
)


@app.post("/callback")
async def handle_callback(request: Request):
    signature = request.headers['X-Line-Signature']

    # get request body as text
    body = await request.body()
    body = body.decode()

    try:
        events = parser.parse(body, signature)
    except InvalidSignatureError:
        raise HTTPException(status_code=400, detail="Invalid signature")

    for event in events:
        if not isinstance(event, MessageEvent):
            continue
        if not isinstance(event.message, TextMessage):
            continue

        response = qa_with_sources_chain({"query": event.message.text})
        print(response['source_documents'])
        # first best source document.
        doc = response['source_documents'][0]

        await line_bot_api.reply_message(
            event.reply_token,
            TextSendMessage(text=response['result']+"\n"+doc.metadata['url'])
        )

    return 'OK'
