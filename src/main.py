import os
from langchain.agents.agent_toolkits import AzureCognitiveServicesToolkit
from langchain.llms import AzureOpenAI
from langchain.agents import initialize_agent, AgentType
from dotenv import load_dotenv

import warnings
warnings.filterwarnings("ignore")

# Dapatkan variabel lingkungan yang telah dimuat
load_dotenv()
AZURE_COGS_KEY = os.getenv("AZURE_COGS_KEY")
AZURE_COGS_ENDPOINT = os.getenv("AZURE_COGS_ENDPOINT")
AZURE_COGS_REGION = os.getenv("AZURE_COGS_REGION")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")

os.environ["AZURE_COGS_KEY"] = AZURE_COGS_KEY
os.environ["AZURE_COGS_ENDPOINT"] = AZURE_COGS_ENDPOINT 
os.environ["AZURE_COGS_REGION"] = "eastus"

toolkit = AzureCognitiveServicesToolkit()

# print([tool.name for tool in toolkit.get_tools()])

llm = AzureOpenAI(azure_deployment="gpt-35-turbo",
                  openai_api_version="2023-05-15",
                  openai_api_key=OPENAI_API_KEY,
                  azure_endpoint=OPENAI_ENDPOINT)
agent = initialize_agent(
    tools=toolkit.get_tools(),
    llm=llm,
    agent=AgentType.STRUCTURED_CHAT_ZERO_SHOT_REACT_DESCRIPTION,
    verbose=False,
)

# agent.run("generate a funny story targeted to children about the following image: "
#     "https://www.vacanzeanimali.it/images/strutture/44910.jpg")

# response = agent.run("berapa jumlah subtotal?" "https://miro.medium.com/v2/resize:fit:700/0*4VS1bGjXxX4p0iy5.png")
image_path = os.path.join(os.getcwd(), "image", "bank_statement.png")
user_question = "berapa jumlah subtotal?"
# response = agent.run(f'{user_question} {image_path}')
response = agent.run(f'"{user_question}" "{image_path}"')
print(response)