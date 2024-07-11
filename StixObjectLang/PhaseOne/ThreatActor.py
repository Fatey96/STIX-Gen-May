import os
from typing import List, Optional
import dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_experimental.tabular_synthetic_data.openai import (
    OPENAI_TEMPLATE,
    create_openai_data_generator,
)
from langchain_experimental.tabular_synthetic_data.prompts import (
    SYNTHETIC_FEW_SHOT_PREFIX,
    SYNTHETIC_FEW_SHOT_SUFFIX,
)
from langchain_community.chat_models import ChatOpenAI

dotenv.load_dotenv()

# Define the ThreatActor schema
class ThreatActor(BaseModel):
    type: str = Field(default="threat-actor")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the threat actor")
    created: str = Field(description="Creation date of the threat actor entry")
    modified: str = Field(description="Last modification date of the threat actor entry")
    name: str = Field(description="Name of the threat actor")
    threat_actor_types: List[str] = Field(description="Types of threat actor")
    description: Optional[str] = Field(default=None, description="Description of the threat actor")
    aliases: Optional[List[str]] = Field(default=None, description="Alternative names for the threat actor")
    first_seen: str = Field(description="Date when the threat actor was first observed")
    last_seen: str = Field(description="Date when the threat actor was last observed")
    roles: Optional[List[str]] = Field(default=None, description="Roles played by the threat actor")
    goals: Optional[List[str]] = Field(default=None, description="Objectives of the threat actor")
    sophistication: Optional[str] = Field(default=None, description="Level of sophistication of the threat actor")
    resource_level: Optional[str] = Field(default=None, description="Resources available to the threat actor")
    primary_motivation: Optional[str] = Field(default=None, description="Main motivation of the threat actor")
    secondary_motivations: Optional[List[str]] = Field(default=None, description="Secondary motivations of the threat actor")
    personal_motivations: Optional[List[str]] = Field(default=None, description="Personal motivations of the threat actor")

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
    llm=ChatOpenAI(temperature=1, model='gpt-4o'),
    prompt=prompt_template,
)

def generate_threat_actor(count: int) -> List[ThreatActor]:
    """
    Generate synthetic threat actors.
    
    Args:
        count (int): Number of threat actors to generate.
    
    Returns:
        List[ThreatActor]: List of generated threat actors.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="threat_actor",
        extra="Choose a unique and unconventional name for each threat actor. Avoid common or typical names.",
        runs=count,
    )
    return synthetic_results