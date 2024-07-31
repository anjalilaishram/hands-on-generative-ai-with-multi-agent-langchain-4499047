# Import necessary libraries and modules
import os
import math
import faiss
import logging
from datetime import datetime, timedelta
from typing import List
from termcolor import colored
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain_experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory

# Setup logging
logging.basicConfig(level=logging.ERROR)

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = ''  # Insert your OpenAI API key here

# Define constants
USER_NAME = "Nayan"  # The name you want to use when interviewing the agent
LLM = ChatOpenAI(max_tokens=1500)  # Can be any LLM you want

# Function to calculate relevance score
def relevance_score_fn(score: float) -> float:
    """Return a similarity score on a scale [0, 1]."""
    return 1.0 - score / math.sqrt(2)

# Function to create a new memory retriever
def create_new_memory_retriever():
    """Create a new vector store retriever unique to the agent."""
    embeddings_model = OpenAIEmbeddings()
    embedding_size = 1536
    index = faiss.IndexFlatL2(embedding_size)
    vectorstore = FAISS(
        embeddings_model.embed_query,
        index,
        InMemoryDocstore({}),
        {},
        relevance_score_fn=relevance_score_fn,
    )
    return TimeWeightedVectorStoreRetriever(
        vectorstore=vectorstore, other_score_keys=["importance"], k=15
    )

# Function to interview an agent
def interview_agent(agent: GenerativeAgent, message: str) -> str:
    """Help the user interact with the agent."""
    new_message = f"{USER_NAME} says {message}"
    return agent.generate_dialogue_response(new_message)[1]

# Function to run a conversation between agents
def run_conversation(agents: List[GenerativeAgent], initial_observation: str) -> None:
    """Runs a conversation between agents."""
    _, observation = agents[1].generate_reaction(initial_observation)
    print(observation)
    max_turns = 3
    turns = 0
    while turns <= max_turns:
        break_dialogue = False
        for agent in agents:
            stay_in_dialogue, observation = agent.generate_dialogue_response(
                observation
            )
            print(observation)
            if not stay_in_dialogue:
                break_dialogue = True
        if break_dialogue:
            break
        turns += 1

# Function to run a competitive trivia night between agents
def run_competitive_trivia(agents: List[GenerativeAgent], questions: List[str]) -> None:
    """Runs a competitive trivia night between agents."""
    for question in questions:
        print(f"Trivia Question: {question}")

        for agent in agents:
            response = agent.generate_dialogue_response(question)[1]
            print(f"{agent.name}'s Answer: {response}")

        print("-" * 40)

# Create Alexis's memory
alexis_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=8,  # we will give this a relatively low number to show how reflection works
)

# Defining the Generative Agent: Alexis
alexis = GenerativeAgent(
    name="Alexis",
    age=30,
    traits="curious, creative writer, world traveler",  # Persistent traits of Alexis
    status="exploring the intersection of technology and storytelling",  # Current status of Alexis
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=alexis_memory,
)

# Add initial observations to Alexis's memory
alexis_observations = [
    "Alexis recalls her morning walk in the park",
    "Alexis feels excited about the new book she started reading",
    "Alexis remembers her conversation with a close friend",
    "Alexis thinks about the painting she saw at the art gallery",
    "Alexis is planning to learn a new recipe for dinner",
    "Alexis is looking forward to her weekend trip",
    "Alexis contemplates her goals for the month."
]

for observation in alexis_observations:
    alexis.memory.add_memory(observation)

# Print Alexis's summary after initial observations
print(alexis.get_summary(force_refresh=True))

# Add daily observations to Alexis's memory
alexis_observations_day = [
    "Alexis starts her day with a refreshing yoga session.",
    "Alexis spends time writing in her journal.",
    "Alexis experiments with a new recipe she found online.",
    "Alexis gets lost in her thoughts while gardening.",
    "Alexis decides to call her grandmother for a heartfelt chat.",
    "Alexis relaxes in the evening by playing her favorite piano pieces.",
]

for observation in alexis_observations_day:
    alexis.memory.add_memory(observation)

# Print Alexis's summary after daily observations
for i, observation in enumerate(alexis_observations_day):
    _, reaction = alexis.generate_reaction(observation)
    print(colored(observation, "green"), reaction)
    if ((i + 1) % len(alexis_observations_day)) == 0:
        print("*" * 40)
        print(
            colored(
                f"After these observations, Alexis's summary is:\n{alexis.get_summary(force_refresh=True)}",
                "blue",
            )
        )
        print("*" * 40)

# Create Jordan's memory
jordan_memory = GenerativeAgentMemory(
    llm=LLM,
    memory_retriever=create_new_memory_retriever(),
    verbose=False,
    reflection_threshold=7,  # Set to illustrate Jordan's reflective capabilities
)

# Defining the Generative Agent: Jordan
jordan = GenerativeAgent(
    name="Jordan",
    age=28,
    traits="tech enthusiast, avid gamer, foodie",  # Persistent traits of Jordan
    status="navigating the world of tech startups",  # Current status of Jordan
    memory_retriever=create_new_memory_retriever(),
    llm=LLM,
    memory=jordan_memory,
)

# Add initial observations to Jordan's memory
jordan_observations_day = [
    "Jordan finished a challenging coding project last night",
    "Jordan won a local gaming tournament over the weekend",
    "Jordan tried a new sushi restaurant and loved it",
    "Jordan read an article about the latest AI advancements",
    "Jordan is planning a meetup with tech enthusiasts",
    "Jordan discovered a bug in his latest app prototype",
    "Jordan booked tickets for a tech conference next month",
    "Jordan feels excited about a potential startup idea",
    "Jordan spent the evening playing video games to unwind",
    "Jordan is considering enrolling in a machine learning course"
]

for observation in jordan_observations_day:
    jordan.memory.add_memory(observation)

# Print Jordan's summary after initial observations
print(jordan.get_summary())

# Run a conversation between Alexis and Jordan
agents = [alexis, jordan]
run_conversation(
    agents,
    "Alexis said: Hey Jordan, I've been exploring how technology influences creativity lately. Since you're into tech, I was wondering if you've seen any interesting intersections in your field?",
)

# Interview agents after their conversation
print(interview_agent(jordan, "How was your conversation with Alexis?"))
print(interview_agent(alexis, "How was your conversation with Jordan?"))

# Run a competitive trivia night between Alexis and Jordan
trivia_questions = [
    "What is the capital city of France?",
    "Who is known as the father of modern computing?",
    "Can you name a famous work of art by Leonardo da Vinci?",
]

run_competitive_trivia(agents, trivia_questions)
