import os
os.environ["OPENAI_API_KEY"] = "sk-proj-6K2dANij8t7gG7FohQxBT3BlbkFJtkFeltl7Ua7K6JuRF6th"

"""# Imports"""

from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.pydantic_v1 import BaseModel
from datetime import datetime
from typing import List, Optional
from langchain_experimental.tabular_synthetic_data.base import SyntheticDataGenerator
from langchain_experimental.tabular_synthetic_data.openai import create_openai_data_generator, OPENAI_TEMPLATE
from langchain_experimental.tabular_synthetic_data.prompts import SYNTHETIC_FEW_SHOT_SUFFIX, SYNTHETIC_FEW_SHOT_PREFIX

"""# Schema for generating Grouping



"""

class IntrusionSet(BaseModel):
    type: str
    spec_version: str
    id: str
    created: str
    modified: str
    name: str
    description: Optional[str] = None
    aliases: Optional[str] = None
    first_seen: Optional[str] = None
    last_seen: Optional[str] = None
    goals: Optional[str] = None
    resource_level: Optional[str] = None
    primary_motivation: Optional[str] = None
    secondary_motivation: Optional[str] = None

"""# Sample Data as example"""

examples = [
    {"example": """Type: intrusion_set, spec_version: 2.1, id: intrusion-set--4e78f46f-a023-4e5f-bc24-71b3ca22ec29, created: 2016-04-06T20:03:48.000Z, modified: 2016-04-06T20:03:48.000Z, name: Bobcat Breakin, description: Incidents usually feature a shared TTP of a bobcat being released within the building containing network access, scaring users to leave their computers without locking them first. Still determining where the threat actors are getting the bobcats., aliases: zookeeper, goals: acquistion-theft"""},
    {"example": """Type: intrusion_set, spec_version: 2.1, id:intrusion-set--da1065ce-972c-4605-8755-9cd1074e3b5a, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, name: APT1, description: APT1 is a single organization of operators that has conducted a cyber espionage campaign against a broad range of victims since at least 2006, first_seen: 2006-06-01T18:13:15.684Z, resource_level: government, primary motivation: organizational-gain, aliases: Comment Crew, Comment Group, shady rat"""},
    {"example": """Type: intrusion_set, spec_version: 2.1, id: intrusion-set--ed69450a-f067-4b51-9ba2-c4616b9a6713, created: 2016-08-08T15:50:10.983Z, modified: 2016-08-08T15:50:10.983Z, name: APT BPP, description: An advanced persistent threat that seeks to disrupt Branistan's election with multiple attacks., aliases: bran-teaser, first_seen: 2016-01-08T12:50:40.123Z, goals: Influence the Branistan election, disrupt the BPP, resource_level: government, primary_motivation: ideology, secondary_motivation: dominance"""},
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

"""# Data Generator"""

synthetic_data_generator = create_openai_data_generator(
    output_schema=IntrusionSet,
    llm=ChatOpenAI(temperature=1,model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

"""# Parameters"""

synthetic_results = synthetic_data_generator.generate(
    subject="IntrusionSet ",
    extra="Choose a unique and unconventional name, description, aliases, first_seen, last_seen, goals, resource_level, primary_motivation, secondary_motivations. Avoid common or typical names.",
    runs=55,
)

len(synthetic_results)

"""# Display Data"""

synthetic_results

"""# Display as a DataFrame"""

import pandas as pd

# Create a list of dictionaries from the objects
synthetic_data = []
for item in synthetic_results:
    synthetic_data.append({
        'type': item.type,
        'name': item.name,
        'description': item.description,
        'aliases': item.aliases,
        'first_seen': item.first_seen,
        'last_seen': item.last_seen,
        'goals': item.goals,
        'resource_level': item.resource_level,
        'primary_motivation': item.primary_motivation,
        'secondary_motivation': item.secondary_motivation
        })

# Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_data)

# Display the DataFrame
print(type(synthetic_df))
synthetic_df

# Save the DataFrame to a CSV file
synthetic_df.to_csv('intrusion_set.csv', index=False)  # index=False prevents adding an extra index column
print("intrusion set data saved to 'intrusion_set.csv'")