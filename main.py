import os
import openai
from langchain.vectorstores import Chroma
from langchain.embeddings.openai import OpenAIEmbeddings
import telebot

openai.api_key = "<YOUR_TOKEN>"
os.environ["OPENAI_API_KEY"] = "<YOUR_TOKEN>"
openai.api_type = "..."
os.environ["OPENAI_API_TYPE"] = "..."
openai.api_base = "<your base address>"
os.environ["OPENAI_API_BASE"] = "<your base address>"
openai.api_version = "2023-07-01-preview"
os.environ["OPENAI_API_VERSION"] = "2023-07-01-preview"
bot = telebot.TeleBot("<your bot token>")


from langchain.agents import AgentType
from langchain.prompts import ChatPromptTemplate
from langchain.agents import initialize_agent
from langchain.chat_models import AzureChatOpenAI
from langchain.memory import ConversationSummaryBufferMemory, ConversationBufferMemory

from langchain.agents import AgentExecutor, Tool, ZeroShotAgent
from langchain.chains import LLMChain
from langchain.llms import OpenAI

llm = AzureChatOpenAI(
            deployment_name="your_model_name",
            openai_api_version="2023-07-01-preview",
            temperature=0,
        )

default_prompt = ChatPromptTemplate.from_template("{input}")
gpt_chain = LLMChain(llm=llm, prompt=default_prompt)

persist_directory = "db/"
embeddings = OpenAIEmbeddings(deployment="<your_embed_model>", chunk_size=3, timeout=60, show_progress_bar=True, retry_min_seconds=15)
db = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
# print(get_responce(input("Enter your prompt: "), my_db))

metadata = []

def get_responce(user_input):
    context_docs = db.similarity_search(user_input)
    # context = context_docs[0].page_content
    context = '\n'.join([doc.page_content for doc in context_docs[:3]])
    global metadata
    metadata = [doc.metadata for doc in context_docs[:3]]
    systemMessage = '''You are an AI assistant that helps people find information based on provided context.
    **Always answer on English**

    ###

    CONTEXT:
    '''
    systemMessage += context
    messages = [{"role": "system", "content": systemMessage},
                {"role": "user", "content": user_input}]

    # Generate the response using the GPT model
    response = openai.ChatCompletion.create(
        engine="<model_name>",
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

tools = [
            Tool(
                name="Vector Search",
                func=get_responce,
                description="Vector Search fuction **as the default request tool**, specialized tool for querying the information about Computer system  sunject. Can get answer about types of systems and other embedding systems. Use by default.",
            ),
            # Tool(
            #     name="GPT request",
            #     func=gpt_chain.run,
            #     description="Serves capable of tasks such as code writing and answering questions without the need for a Google search. Use when you dont know actual answer and Vector Search tool doesnt helps you.",
            # ),
        ]



prefix = """Have a conversation with a human, answering the following questions as best you can. You have access to the following tools:"""
suffix = """Begin!"

{chat_history}
Question: {input}
{agent_scratchpad}
**As Action input use the prompt language**
"""

prompt = ZeroShotAgent.create_prompt(
    tools,
    prefix=prefix,
    suffix=suffix,
    input_variables=["input", "chat_history", "agent_scratchpad"],
)

def create_agent_chain():
    memory = ConversationSummaryBufferMemory(llm=llm, max_token_limit=5000, memory_key="chat_history",)
    llm_chain = LLMChain(llm=llm, prompt=prompt)
    agent = ZeroShotAgent(llm_chain=llm_chain, tools=tools, verbose=True)
    agent_chain = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=tools,
        verbose=True,
        memory=memory,
        handle_parsing_errors=True,
        max_iterations=1,
        early_stopping_method="generate",
    )
    return agent_chain
agent_chain = create_agent_chain()

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
        global metadata, agent_chain
        try:
            # answer, metadata = get_responce(message.text, my_db)
            answer = agent_chain.run(message.text)
            bot.reply_to(message, answer)
            # printing metadata like document name and page for better answer
            text = "sources: \n"
            i = 0
            for content in metadata:
                i += 1
                text += f"{i})" + content['source'].split('/')[-1] + f" - page: {content['page']}" + "\n"
            bot.reply_to(message, text)
        except Exception as ex:
            print("Exception: - ", ex)
            bot.reply_to(message, "Getting Error, trying one more time...")

            agent_chain = create_agent_chain()
            try:
                answer = agent_chain.run(message.text)
                bot.reply_to(message, answer)
                text = "sources: \n"
                i = 0
                for content in metadata:
                    i += 1
                    text += f"{i})" + content['source'].split('/')[-1] + f" - page: {content['page']}" + "\n"
                bot.reply_to(message, text)
            except Exception as ex:
                print("Exception on retry: ", ex)
                bot.reply_to(message, "Your request could not be processed, please try again later")

    else:
        print(f"{user_tg} tried to use bot.")
        print("His command - ", message.text)
        bot.reply_to(message, "Sorry, you do not have access to this bot. To apply, write to -> ")

print("Running...")
bot.infinity_polling()
