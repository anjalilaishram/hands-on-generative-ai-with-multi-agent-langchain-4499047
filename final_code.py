# Import necessary libraries and modules
import os
import math
import faiss
import logging
from datetime import datetime, timedelta
from typing import List, Callable
from termcolor import colored
from langchain.chat_models import ChatOpenAI
from langchain.docstore import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.retrievers import TimeWeightedVectorStoreRetriever
from langchain.vectorstores import FAISS
from langchain_experimental.generative_agents import GenerativeAgent, GenerativeAgentMemory
from langchain.memory import ConversationBufferMemory
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain.output_parsers import RegexParser
from langchain.prompts import PromptTemplate
from langchain.agents import AgentType, initialize_agent, load_tools
import tenacity
import numpy as np

# Setup logging
logging.basicConfig(level=logging.ERROR)

# Set the OpenAI API key
os.environ["OPENAI_API_KEY"] = 'your_openai_api_key_here'  # Insert your OpenAI API key here

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

# Define the DialogueAgent class
class DialogueAgent:
    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI) -> None:
        self.name = name
        self.system_message = system_message
        self.model = model
        self.prefix = f"{self.name}: "
        self.reset()

    def reset(self):
        self.message_history = ["Here is the conversation so far."]

    def send(self) -> str:
        """
        Applies the chat model to the message history and returns the message string
        """
        message = self.model(
            [
                self.system_message,
                HumanMessage(content="\n".join(self.message_history + [self.prefix])),
            ]
        )
        return message.content

    def receive(self, name: str, message: str) -> None:
        """
        Concatenates {message} spoken by {name} into message history
        """
        self.message_history.append(f"{name}: {message}")

# Define the DialogueSimulator class
class DialogueSimulator:
    def __init__(self, agents: List[DialogueAgent], selection_function: Callable[[int, List[DialogueAgent]], int]) -> None:
        self.agents = agents
        self._step = 0
        self.select_next_speaker = selection_function

    def reset(self):
        for agent in self.agents:
            agent.reset()

    def inject(self, name: str, message: str):
        """
        Initiates the conversation with a {message} from {name}
        """
        for agent in self.agents:
            agent.receive(name, message)
        self._step += 1

    def step(self) -> tuple[str, str]:
        # 1. choose the next speaker
        speaker_idx = self.select_next_speaker(self._step, self.agents)
        speaker = self.agents[speaker_idx]

        # 2. next speaker sends message
        message = speaker.send()

        # 3. everyone receives message
        for receiver in self.agents:
            receiver.receive(speaker.name, message)

        # 4. increment time
        self._step += 1

        return speaker.name, message

# Define the BiddingDialogueAgent class
class BiddingDialogueAgent(DialogueAgent):
    def __init__(self, name, system_message: SystemMessage, bidding_template: PromptTemplate, model: ChatOpenAI) -> None:
        super().__init__(name, system_message, model)
        self.bidding_template = bidding_template

    def bid(self) -> str:
        """
        Asks the chat model to output a bid to speak
        """
        prompt = PromptTemplate(
            input_variables=["message_history", "recent_message"],
            template=self.bidding_template,
        ).format(
            message_history="\n".join(self.message_history),
            recent_message=self.message_history[-1],
        )
        bid_string = self.model([SystemMessage(content=prompt)]).content
        return bid_string

# Helper functions to generate character descriptions and messages
def generate_character_description(character_name):
    character_specifier_prompt = [
        player_descriptor_system_message,
        HumanMessage(
            content=f"""Here is the topic for the startup pitch to investors Sandra and Daniel: {topic}.
            Please reply with a creative description of {character_name}, in {word_limit} words or less, that emphasizes their personalities.
            Speak directly to {character_name}.
            Do not add anything else."""
        ),
    ]
    character_description = ChatOpenAI(temperature=0.6)(character_specifier_prompt).content
    return character_description

def generate_character_header(character_name, character_description):
    return f"""Here is the topic for the startup pitch to investors Sandra and Daniel: {topic}.
Your name is {character_name}.
Your description is as follows: {character_description}
Your topic is: {topic}.
"""

def generate_character_system_message(character_name, character_header):
    return SystemMessage(
        content=(
            f"""{character_header}
You will speak in the style of {character_name}, and exaggerate their personality RESPONDING in under 450 characters.
You will come up with creative ideas related to {topic}.
Do not say the same things over and over again.
Speak in the first person from the perspective of {character_name}
ONLY SPEAK FOR YOURSELF WHO IS {character_name} AND NOT OTHER CHARACTERS FROM {', '.join(character_names)}
For describing your own body movements, wrap your description in '*'.
Do not change roles!
Do not speak from the perspective of anyone else.
Speak only from the perspective of {character_name}.
Stop speaking the moment you finish speaking from your perspective.
Never forget to keep your response to {word_limit} words!
Do not add anything else.
"""
        )
    )

# Define the BidOutputParser class
class BidOutputParser(RegexParser):
    def get_format_instructions(self) -> str:
        return "Your response should be an integer delimited by angled brackets, like this: <int>."

bid_parser = BidOutputParser(
    regex=r"<(\d+)>", output_keys=["bid"], default_output_key="bid"
)

@tenacity.retry(
    stop=tenacity.stop_after_attempt(2),
    wait=tenacity.wait_none(),  # No waiting time between retries
    retry=tenacity.retry_if_exception_type(ValueError),
    before_sleep=lambda retry_state: print(
        f"ValueError occurred: {retry_state.outcome.exception()}, retrying..."
    ),
    retry_error_callback=lambda retry_state: 0,
)  # Default value when all retries are exhausted
def ask_for_bid(agent) -> str:
    """
    Ask for agent bid and parses the bid into the correct format.
    """
    bid_string = agent.bid()
    bid = int(bid_parser.parse(bid_string)["bid"])
    return bid

def generate_character_bidding_template(character_header):
    bidding_template = f"""{character_header}

`{{message_history}}`

On the scale of 1 to 10, where 1 is least important to the startup pitch and 10 is extremely important and contribute, rank your recent message based on the context. Make sure to be very thorough in your ranking and only rank stuff that is important higher.

`{{recent_message}}`

{bid_parser.get_format_instructions()}
Do nothing else.
    """
    return bidding_template

# Define the speaker selection function
def select_next_speaker(step: int, agents: List[DialogueAgent]) -> int:
    bids = []
    for agent in agents:
        bid = ask_for_bid(agent)
        bids.append(bid)

    # randomly select among multiple agents with the same bid
    max_value = np.max(bids)
    max_indices = np.where(bids == max_value)[0]
    idx = np.random.choice(max_indices)

    print("Bids:")
    for i, (bid, agent) in enumerate(zip(bids, agents)):
        print(f"\t{agent.name} bid: {bid}")
        if i == idx:
            selected_name = agent.name
    print(f"Selected: {selected_name}")
    print("\n")
    return idx

# Define participants and debate topic
character_names = ["CTO", "CMO", "CEO", "Investor-Daniel", "Investor-Sandra"]
topic = "Startup pitch on startup focused on energy drinks with no caffeine"
word_limit = 15

# Define the simulation
game_description = f"""Here is the topic for the startup pitch to investors Sandra and Daniel: {topic}.
The participants are: {', '.join(character_names)}."""

# Generate character descriptions, headers, and system messages
player_descriptor_system_message = SystemMessage(content="You can add detail to the description of each participant")
character_descriptions = [generate_character_description(character_name) for character_name in character_names]
character_headers = [generate_character_header(character_name, character_description) for character_name, character_description in zip(character_names, character_descriptions)]
character_system_messages = [generate_character_system_message(character_name, character_header) for character_name, character_header in zip(character_names, character_headers)]

# Generate character bidding templates
character_bidding_templates = [generate_character_bidding_template(character_header) for character_header in character_headers]

# Create Bidding Dialogue Agents for each character
characters = []
model = ChatOpenAI(temperature=0.4)
for character_name, character_system_message, bidding_template in zip(character_names, character_system_messages, character_bidding_templates):
    characters.append(BiddingDialogueAgent(name=character_name, system_message=character_system_message, model=model, bidding_template=bidding_template))

# Run the simulation
max_iters = 10
n = 0

simulator = DialogueSimulator(agents=characters, selection_function=select_next_speaker)
simulator.reset()

first_message = "CEO, CMO, CTO You can now start pitching your ideas to our investor Sandra and Daniel"
simulator.inject("Moderator", first_message)
print(f"(Moderator): {first_message}")
print("\n")

while n < max_iters:
    name, message = simulator.step()
    print(f"({name}): {message}")
    print("\n")
    n += 1

# Define the DialogueAgentWithTools class
class DialogueAgentWithTools(DialogueAgent):
    def __init__(self, name: str, system_message: SystemMessage, model: ChatOpenAI, tool_names: List[str], **tool_kwargs) -> None:
        super().__init__(name, system_message, model)
        self.tools = load_tools(tool_names, **tool_kwargs)

    def send(self) -> str:
        """
        Applies the chat model to the message history and returns the message string
        """
        agent_chain = initialize_agent(
            self.tools,
            self.model,
            agent=AgentType.CHAT_CONVERSATIONAL_REACT_DESCRIPTION,
            verbose=True,
            memory=ConversationBufferMemory(
                memory_key="chat_history", return_messages=True
            ),
        )
        message = AIMessage(
            content=agent_chain.run(
                input="\n".join(
                    [self.system_message.content] + self.message_history + [self.prefix]
                )
            )
        )

        return message.content

# Agent Descriptions
agent_descriptions = {
    "Alex": "Alex is a strong advocate for remote work, emphasizing its flexibility and productivity benefits.",
    "Jordan": "Jordan is skeptical about remote work, focusing on potential downsides like reduced team interaction."
}

# Generate System Messages
def generate_system_message(name, description):
    return f"""Your name is {name}.
          Your description is as follows: {description}

          Your goal is to persuade your conversation partner of your point of view.

          DO look up information with your tool to refute your partner's claims.
          DO cite your sources.

          DO NOT fabricate fake citations.
          DO NOT cite any source that you did not look up.

          Do not add anything else.

          Stop speaking the moment you finish speaking from your perspective.
          """

agent_system_messages = {name: generate_system_message(name, description) for name, description in agent_descriptions.items()}

# Topic Specification
specified_topic = "The Impact of Remote Work on Employee Productivity"

# Agent Setup
agents_with_tools = [
    DialogueAgentWithTools(
        name=name,
        system_message=SystemMessage(content=system_message),
        model=ChatOpenAI(model_name="gpt-4", temperature=0.2),
        tool_names= ["arxiv", "ddg-search", "wikipedia"],
        top_k_results=2,
    ) for name, system_message in agent_system_messages.items()
]

# Speaker Selection Function for agents with tools
def select_next_speaker_with_tools(step, agents_with_tools):
    return step % len(agents_with_tools)

# Running the Simulation for agents with tools
max_iters_with_tools = 4
simulator_with_tools = DialogueSimulator(agents=agents_with_tools, selection_function=select_next_speaker_with_tools)
simulator_with_tools.reset()
simulator_with_tools.inject("Moderator", specified_topic)

for _ in range(max_iters_with_tools):
    name, message = simulator_with_tools.step()
    print(f"({name}): {message}")
