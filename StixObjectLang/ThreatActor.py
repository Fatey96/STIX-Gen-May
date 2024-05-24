import os
os.environ["OPENAI_API_KEY"] = "sk-proj-6K2dANij8t7gG7FohQxBT3BlbkFJtkFeltl7Ua7K6JuRF6th"

"""# Imports"""

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import BaseModel
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_SUFFIX, SYNTHETIC_FEW_SHOT_PREFIX
from typing import List, Optional

# Define the ThreatActor schema
class ThreatActor(BaseModel):
    type: str
    spec_version: str
    id: str
    created: str
    modified: str
    name: str
    threat_actor_types: Optional[List[str]]
    description: Optional[str] = None
    aliases: Optional[List[str]] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    roles: Optional[List[str]] = None
    goals: Optional[List[str]] = None
    sophistication: Optional[str] = None
    resource_level: Optional[str] = None
    primary_motivation: Optional[str] = None
    secondary_motivations: Optional[List[str]] = None
    personal_motivations: Optional[List[str]] = None


"""# Sample Data as example"""

examples = [
    {"example": """Type: threat-actor, Name: Silent Griffin, Threat Actor Types: nation-state, Description: They are a sophisticated nation-state actor believed to be operating under the directive of the government of Country X. Its activities are primarily focused on espionage and gathering intelligence from foreign governments and corporations, Aliases: Griffin Shadow, State Griffin, Roles: spy, Goals: Collect intelligence, Disrupt foreign infrastructure, Sophistication: expert, Resource Level: government, Primary Motivation: ideology"""},
    {"example": """Type: threat-actor, Name: CyberMafia, Threat Actor Types: crime-syndicate, Description: They are an organized crime group known for its involvement in various cybercrimes, including financial fraud, ransomware attacks, and cyber extortion, Aliases: Digital Mobsters, NetMafia, Roles: executor, planner, Goals: Generate profit, Launder money, Sophistication: advanced, Resource Level: organization, Primary Motivation: financial-gain"""},
    {"example": """Type: threat-actor, Name: Digital Rebels, Threat Actor Types: hacktivist, Description: They are a collective of hacktivists known for launching cyber attacks against organizations they perceive as unethical or corrupt. Their operations often aim to expose secrets or disrupt operations to make a political statement, Aliases: NetActivists, Rebel Hackers, Roles: activist, Goals: Expose corruption, Promote transparency, Sophistication: intermediate, Resource Level: group, Primary Motivation: ideology"""},
    {"example": """Type: threat-actor, Name: John Doe, Threat Actor Types: insider, Description: They are a disgruntled employee within Corporation X, has been identified as an insider threat, leveraging his access to sensitive information to inflict harm on the company, Roles: insider, Goals: Steal data, Sabotage company operations, Sophistication: basic, Resource Level: individual, Primary Motivation: personal-satisfaction"""},
    {"example": """Type: threat-actor, Name: Shadow Faction, Threat Actor Types: terrorist, Description: They are a terrorist cyber unit that employs cyber attacks to advance their extremist ideology and cause disruption or harm to their perceived enemies, Aliases: Dark Unit, Terror Byte, Roles: cyber-terrorist, Goals: Disrupt critical infrastructure, Spread fear, Sophistication: advanced, Resource Level: group, Primary Motivation: ideology"""},
    {"example": """Type: threat-actor, Name: Phantom Hawk, Threat Actor Types: state-sponsored, Description: A sophisticated state-funded threat actor specializing in cyber espionage operations against critical infrastructure and government entities, Aliases: Dark Vortex, Roles: developer, sponsor, Goals: Disrupt government operations, Intellectual property theft, Sabotage critical infrastructure, Sophistication: expert, Resource Level: government, Primary Motivation: national-security"""},
    {"example": """Type: threat-actor, Name: The Shadow Network, Threat Actor Types: hacktivist, Description: A decentralized collective of hacktivists known for targeting corporations and governments engaged in activities the group deems unethical, Roles: activist, Goals: Expose corruption, Disrupt unethical organizations, Promote social change, Sophistication: intermediate, Resource Level: individual, Primary Motivation: ideological"""},
    {"example": """Type: threat-actor, Name: Serpent Fang, Threat Actor Types: crime-syndicate, Description: A financially motivated ransomware group known for targeting healthcare and educational institutions. Specializes in double extortion schemes, Aliases: Venom Strike, Roles: developer, operator, Goals: Extortion, Data theft for leverage, Sophistication: expert, Resource Level: organization, Primary Motivation: personal-gain"""},
    {"example": """Type: threat-actor, Name: Disco Team Threat Actor Group, Threat Actor Types: crime-syndicate, Description: This organized threat actor group operates to create profit from all types of crime, Aliases: Equipo del Discoteca, Roles: agent, Goals: Steal Credit Card Information, Sophistication: expert, Resource Level: organization, Primary Motivation: personal-gain"""},
    {"example": """Type: threat-actor, Name: Sarah Smith, Threat Actor Types: insider-threat, Description: They are a disgruntled former employee of Corporation X, seeks to cause harm to the company after being terminated.  She has retained some access to company systems, Roles: insider, Goals: Sabotage company operations, Leak sensitive data, Sophistication: intermediate, Resource Level: individual, Primary Motivation: revenge """},
]

"""# Prompt Template for GPT-4"""

OPENAI_TEMPLATE = PromptTemplate(input_variables=["example"], template="{example}")

prompt_template = FewShotPromptTemplate(
    prefix=SYNTHETIC_FEW_SHOT_PREFIX,
    examples=examples,
    suffix=SYNTHETIC_FEW_SHOT_SUFFIX,
    input_variables=["subject", "extra"],
    example_prompt=OPENAI_TEMPLATE,
)

# Create a LangChain data generator
synthetic_data_generator = create_openai_data_generator(
    output_schema=ThreatActor,
    llm=ChatOpenAI(temperature=1, model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

def generate_threat_actors(count):
    synthetic_results = synthetic_data_generator.generate(
        subject="threat_actor",
        extra="Choose a unique and unconventional name for each threat actor. Avoid common or typical names.",
        runs=count,
    )

    return synthetic_results

