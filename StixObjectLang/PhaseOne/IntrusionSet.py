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

# Define the IntrusionSet schema
class IntrusionSet(BaseModel):
    type: str = Field(default="intrusion-set")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the intrusion set")
    created: str = Field(description="Creation date of the intrusion set entry")
    modified: str = Field(description="Last modification date of the intrusion set entry")
    name: str = Field(description="Name of the intrusion set")
    description: Optional[str] = Field(default=None, description="Description of the intrusion set")
    aliases: Optional[List[str]] = Field(default=None, description="Alternative names for the intrusion set")
    first_seen: Optional[str] = Field(default=None, description="Date when the intrusion set was first observed")
    last_seen: Optional[str] = Field(default=None, description="Date when the intrusion set was last observed")
    goals: Optional[List[str]] = Field(default=None, description="Goals of the intrusion set")
    resource_level: Optional[str] = Field(default=None, description="Level of resources available to the intrusion set")
    primary_motivation: Optional[str] = Field(default=None, description="Primary motivation of the intrusion set")
    secondary_motivations: Optional[List[str]] = Field(default=None, description="Secondary motivations of the intrusion set")

examples = [
    {"example": """Type: intrusion-set, spec_version: 2.1, id: intrusion-set--4e78f46f-a023-4e5f-bc24-71b3ca22ec29, created: 2016-04-06T20:03:48.000Z, modified: 2016-04-06T20:03:48.000Z, name: Bobcat Breakin, description: Incidents usually feature a shared TTP of a bobcat being released within the building containing network access, scaring users to leave their computers without locking them first. Still determining where the threat actors are getting the bobcats., aliases: zookeeper, goals: acquistion-theft"""},
    {"example": """Type: intrusion-set, spec_version: 2.1, id:intrusion-set--da1065ce-972c-4605-8755-9cd1074e3b5a, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, name: APT1, description: APT1 is a single organization of operators that has conducted a cyber espionage campaign against a broad range of victims since at least 2006, first_seen: 2006-06-01T18:13:15.684Z, resource_level: government, primary motivation: organizational-gain, aliases: Comment Crew, Comment Group, shady rat"""},
    {"example": """Type: intrusion-set, spec_version: 2.1, id: intrusion-set--ed69450a-f067-4b51-9ba2-c4616b9a6713, created: 2016-08-08T15:50:10.983Z, modified: 2016-08-08T15:50:10.983Z, name: APT BPP, description: An advanced persistent threat that seeks to disrupt Branistan's election with multiple attacks., aliases: bran-teaser, first_seen: 2016-01-08T12:50:40.123Z, goals: Influence the Branistan election, disrupt the BPP, resource_level: government, primary_motivation: ideology, secondary_motivation: dominance"""},
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
    output_schema=IntrusionSet,
    llm=ChatOpenAI(temperature=1, model='gpt-4o'),
    prompt=prompt_template,
)

def generate_intrusion_set(count: int) -> List[IntrusionSet]:
    """
    Generate synthetic intrusion set entries.
    
    Args:
        count (int): Number of intrusion set entries to generate.
    
    Returns:
        List[IntrusionSet]: List of generated intrusion set entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="intrusion_set",
        extra="Create diverse and unique intrusion sets with unconventional names, descriptions, aliases, goals, and motivations. Ensure a mix of different resource levels and time frames.",
        runs=count,
    )
    return synthetic_results