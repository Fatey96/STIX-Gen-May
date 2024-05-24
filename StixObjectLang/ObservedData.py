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

"""# Schema for generating ObservedData

"""

class ObservedData(BaseModel):
    type: str
    spec_version: str
    id: str
    created: str
    modified: str
    first_observed: str
    last_observed: str
    number_observed: int
    objects: List[str]
    object_refs: List[str]
    created_by_ref: Optional[str] = None
    object_marking_refs: Optional[List[str]] = None
    revoked: Optional[str] = None
    confidence: Optional[str] = None
    labels: Optional[List[str]] = None
    lang: Optional[str] = None
    granular_markings: Optional[List[str]] = None
    extensions: Optional[List[str]] = None

"""# Sample Data as example"""

examples = [
   {"example": """Type: observed-data, First_observed: '2023-01-01T00:00:00Z', Last_observed: '2023-01-01T01:00:00Z', Number_observed: 10, Objects: ['network traffic', 'unusual port access'], Object_refs: ['obj001', 'obj002'], Confidence: high"""},
   {"example": """Type: observed-data, First_observed: '2023-02-15T15:00:00Z', Last_observed: '2023-02-15T15:30:00Z', Number_observed: 5, Objects: ['malware download', 'infected machine'], Object_refs: ['obj003', 'obj004'], Confidence: medium"""},
   {"example": """Type: observed-data, First_observed: '2023-03-10T10:00:00Z', Last_observed: '2023-03-10T11:00:00Z', Number_observed: 20, Objects: ['phishing email', 'credential theft'], Object_refs: ['obj005', 'obj006'], Confidence: high"""},
   {"example": """Type: observed-data, First_observed: '2023-04-05T09:00:00Z', Last_observed: '2023-04-05T09:10:00Z', Number_observed: 2, Objects: ['data exfiltration', 'external drive access'], Object_refs: ['obj007', 'obj008'], Created_by_ref: 'securityTeam001', Confidence: high"""},
   {"example": """Type: observed-data, First_observed: '2023-05-20T20:00:00Z', Last_observed: '2023-05-20T21:00:00Z', Number_observed: 15, Objects: ['DDoS attack', 'server overload'], Object_refs: ['obj009', 'obj010'], Confidence: medium"""},
   {"example": """Type: observed-data, First_observed: '2023-06-15T12:00:00Z', Last_observed: '2023-06-15T12:05:00Z', Number_observed: 1, Objects: ['unauthorized login attempt', 'security breach'], Object_refs: ['obj011', 'obj012'], Confidence: low"""},
   {"example": """Type: observed-data, First_observed: '2023-07-01T18:00:00Z', Last_observed: '2023-07-01T18:30:00Z', Number_observed: 25, Objects: ['suspicious file creation', 'ransomware activation'], Object_refs: ['obj013', 'obj014'], Confidence: high"""},
   {"example": """Type: observed-data, First_observed: '2023-08-23T08:00:00Z', Last_observed: '2023-08-23T08:05:00Z', Number_observed: 3, Objects: ['SQL injection attempt', 'database vulnerability exploit'], Object_refs: ['obj015', 'obj016'], Confidence: high"""},
   {"example": """Type: observed-data, First_observed: '2023-09-10T16:00:00Z', Last_observed: '2023-09-10T16:30:00Z', Number_observed: 8, Objects: ['credit card skimming', 'POS system breach'], Object_refs: ['obj017', 'obj018'], Created_by_ref: 'fraudAnalysisTeam002', Confidence: medium"""}
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
    output_schema=ObservedData,
    llm=ChatOpenAI(temperature=1,model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

"""# Parameters"""

synthetic_results = synthetic_data_generator.generate(
    subject="ObservedData",
    extra="Choose a unique and unconventional type for every ObservedData. Make the object and Object_refs realistic.",
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
        'spec_version': item.spec_version,
        'id': item.id,
        'created': item.created,
        'modified': item.modified,
        'first_observed': item.first_observed,
        'last_observed': item.last_observed,
        'number_observed': item.number_observed,
        'objects': item.objects,
        'object_refs': item.object_refs,
    })

# Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_data)

# Display the DataFrame
print(type(synthetic_df))
synthetic_df

# Save the DataFrame to a CSV file
synthetic_df.to_csv('observed_data_data.csv', index=False)  # index=False prevents adding an extra index column
print("observed_data data saved to 'observed_data_data.csv'")

