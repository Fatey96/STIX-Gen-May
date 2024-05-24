
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

class Indicator(BaseModel):
    type: str
    spec_version: str
    id: str
    created: str
    modified: str
    pattern: str
    pattern_type: str
    valid_from: str
    valid_until: Optional[str] = None
    kill_chain_phases: Optional[str] = None
    pattern_version: Optional[str] = None
    name: Optional[str] = None
    description: Optional[str] = None
    created_by_ref: Optional[str] = None
    indicator_types: Optional[str] = None

"""# Sample Data as example"""

examples = [
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--8e2e2d2b-17d4-4cbf-938f-98ee46b3cd3f, created: 2016-04-06T20:03:48.000Z, modified: 2016-04-06T20:03:48.000Z, name: Poison Ivy malware, description: This file is part of Poision Ivy, pattern: file:hashes.'SHA-256' = '4bac27393bdd9777ce02453256c5577cd02275510b2227f473d03f533924f877', pattern_type: stix, valid_from: 2016-01-01T00:00:00Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--e8094b09-7df4-4b13-b207-1e27af3c4bde, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: ipv4-addr:value = '219.76.208.163', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--329ae6e9-25bd-49e8-89d1-aae4ca52e4a7, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: domain-name:value = 'www.webserver.dynssl.com' OR ipv4-addr:value = '113.10.246.30' OR ipv4-addr:value = '219.90.112.203' OR ipv4-addr:value = '75.126.95.138' OR ipv4-addr:value = '219.90.112.197' OR ipv4-addr:value = '202.65.222.45', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id:indicator--54e1e351-fec0-41a4-b62c-d7f86101e241, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: domain-name:value = 'www.webserver.freetcp.com' OR ipv4-addr:value = '113.10.246.30' OR ipv4-addr:value = '219.90.112.203' OR ipv4-addr:value = '202.65.220.64' OR ipv4-addr:value = '75.126.95.138' OR ipv4-addr:value = '219.90.112.197' OR ipv4-addr:value = '202.65.222.45', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--2e59f00b-0986-437e-9ebd-e0d61900d688, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: domain-name:value = 'www.webserver.fartit.com' OR ipv4-addr:value = '113.10.246.30' OR ipv4-addr:value = '219.90.112.203' OR ipv4-addr:value = '202.65.220.64' OR ipv4-addr:value = '75.126.95.138', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--8da68996-f175-4ae0-bd74-aad4913873b8, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z, pattern: domain-name:value = 'microsofta.byinter.net' OR domain-name:value = 'microsoftb.byinter.net' OR domain-name:value = 'microsoftc.byinter.net' OR domain-name:value = 'microsofte.byinter.net' OR ipv4-addr:value = '113.10.246.30' OR ipv4-addr:value = '219.90.112.203' OR ipv4-addr:value = '202.65.220.64' OR ipv4-addr:value = '75.126.95.138' OR ipv4-addr:value = '219.90.112.197' OR ipv4-addr:value = '202.65.222.45' OR ipv4-addr:value = '98.126.148.114', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""},
    {"example": """Type: indicator, spec_version: 2.1, id: indicator--4e11b23f-732b-418e-b786-4dbf65459d50, created: 2015-05-15T09:12:16.432Z, modified: 2015-05-15T09:12:16.432Z", pattern: domain-name:value = 'nkr.iphone.qpoe.com' OR ipv4-addr:value = '180.210.206.96' OR ipv4-addr:value = '101.78.151.179', pattern_type: stix, valid_from: 2015-05-15T09:12:16.432678Z"""}
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
    output_schema=Indicator,
    llm=ChatOpenAI(temperature=1,model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

"""# Parameters"""

synthetic_results = synthetic_data_generator.generate(
    subject="Indicator",
    extra="Choose a unique and unconventional type, name, description, indicator_types, pattern, pattern_type, pattern_version, valid_from, valid_until, kill_chain_phases for each Indicator. Avoid common or typical names.",
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
        'indicator_types': item.indicator_types,
        'pattern': item.pattern,
        'pattern_type': item.pattern_type,
        'pattern_version': item.pattern_version,
        'valid_from': item.valid_from,
        'valid_until': item.valid_until,
        'kill_chain_phases': item.kill_chain_phases
        })

# Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_data)

# Display the DataFrame
print(type(synthetic_df))
synthetic_df

# Save the DataFrame to a CSV file
synthetic_df.to_csv('indicator_data.csv', index=False)  # index=False prevents adding an extra index column
print("Indicator data saved to 'indicator_data.csv'")