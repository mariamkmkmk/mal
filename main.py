import os
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import telebot

openai.api_key = "..."
os.environ["OPENAI_API_KEY"] = "..."
openai.api_type = "..."
os.environ["OPENAI_API_TYPE"] = "..."
openai.api_base = "..."
os.environ["OPENAI_API_BASE"] = "..."
openai.api_version = "2023-07-01-preview"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
bot = telebot.TeleBot("<your token>")

def get_responce(user_input: str, db):
    context_docs = db.similarity_search(user_input)
    context = context_docs[0].page_content
    systemMessage = '''You are an AI assistant that helps people find information based on provided context.
    Always answer in the same language as the question.

    ###

    CONTEXT:
    '''
    systemMessage += context
    messages = [{"role": "system", "content": systemMessage},
                {"role": "user", "content": user_input}]

    # Generate the response using the GPT model
    response = openai.ChatCompletion.create(
        engine="<your-engine>",
        messages=messages,
        temperature=0,
        max_tokens=1000,
        top_p=0.95,
        frequency_penalty=0,
        presence_penalty=0,
        stop=None)
    # print("SYSTEM PROMPT: ", systemMessage)
    # print()
    # Step 4: Return the generated response
    return response['choices'][0]['message']['content']

persist_directory = "db/"
embeddings = OpenAIEmbeddings(deployment="<your-model>", chunk_size=3, timeout=60, show_progress_bar=True, retry_min_seconds=15)
my_db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# print(get_responce(input("Enter your prompt: "), my_db))

@bot.message_handler(commands=['start', 'help'])
def send_welcome(message):
    bot.reply_to(message, "Hi, I'm CoSy, your Computer Systems tutor. To ask a question, just send me any message. I can generate a response in about 30 sec")

# message.from_user.username

alpha_testers = []
@bot.message_handler(func=lambda message: True)
def echo_all(message):
    user_tg = message.from_user.username
    if any(users in str(user_tg) for users in alpha_testers):
        print("Get responce from ", user_tg)
        print("inputed message: ", message.text)
        try:
            response = get_responce(message.text, my_db)
            bot.reply_to(message, response)
        except:
            bot.reply_to(message, "Getting Error")
    else:
        print(f"{user_tg} tried to use bot.")
        print("His command - ", message.text)
        bot.reply_to(message, "Sorry, you do not have access to this bot. To apply, write to -> ")

print("Running...")
bot.infinity_polling()