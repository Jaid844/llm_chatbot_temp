import cv2
from langchain.tools import BaseTool
from langchain.chat_models import ChatOpenAI
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
from langchain.agents import initialize_agent

from langchain.agents import Tool



from utillis import *
#from langchain.tools import DuckDuckGoSearchTool

model = SentenceTransformer('all-MiniLM-L6-v2')

#search=DuckDuckGoSearchTool()
pinecone.init(api_key='fd9453a9-749a-4400-9673-053edbbe70a7', environment='gcp-starter')
index = pinecone.Index('prototype')

# Read the image




class image_loader(BaseTool):
    name = "Image loader"
    description = "use this tool when you need to load image from user request"
    def  _run(self,image_path, image_directory='downloads') :
        image = cv2.imread(image_path)
        if image is not None:
            # Display the image (optional)
            cv2.imshow('Image', image)
            cv2.waitKey(5000)
            cv2.destroyAllWindows()

    async def _arun(self, image_path):
        raise NotImplementedError("This tool does not support async")

#class search_engine(BaseTool):
#    name="Search Engine"
#    decription = "useful for when you need to answer questions about current events. You should ask targeted questions"
#
#    def _run(self,input):
#        result=search.run(input)
#        return result
#
#    async def _arun(self,input):
#        raise NotImplementedError("This tool does not support async")

llm = ChatOpenAI(model_name="gpt-3.5-turbo", openai_api_key="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")
# initialize conversational memory
conversational_memory = ConversationBufferWindowMemory(
    memory_key='chat_history',
    k=3,
    return_messages=True,
     input_key='input', output_key="output"
)


# initialize agent with tools


PREFIX = '''Answer the question as truthfully as possible using the provided context, 
and if the answer is not contained within the text below, say 'I don't know
'''
FORMAT_INSTRUCTIONS = """To use a tool, please use the following format:
'''
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action 
Observation: If the Escape key is pressed ,then in your observation done

'''

When  you answered the question with provided context just write to user in the form of chat

Thought: Do I need to use a tool? No
AI: [Answer the question as truthfully as possible ]
'''
"""
SUFFIX = '''

Begin! 

Previous conversation history:
{chat_history}

Instructions: {input}
{agent_scratchpad}
'''


tools=[image_loader()]


agent = initialize_agent(
    tools=tools,
    llm=llm,
    agent="chat-conversational-react-description",
    verbose=True,
    return_intermediate_steps=True,
    memory=conversational_memory,
    agent_kwargs={
        'prefix': PREFIX,
        'format_instructions': FORMAT_INSTRUCTIONS,
        'suffix': SUFFIX
    }
)




response = agent(
    {
        "input": "what is urn equation"

    }
)

