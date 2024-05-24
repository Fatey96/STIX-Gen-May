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

"""# Schema for generating Opinion

"""

class Opinion(BaseModel):
    type: str
    spec_version: str
    id: str
    created: str
    modified: str
    opinion: str
    authors: List[str]
    explanation: str
    object_refs: List[str]
    created_by_ref: Optional[str] = None
    object_marking_refs: Optional[List[str]] = None
    revoked: Optional[str] = None
    confidence: Optional[str] = None
    labels: Optional[List[str]] = None
    external_references: Optional[List[str]] = None
    lang: Optional[str] = None
    granular_markings: Optional[List[str]] = None
    extensions: Optional[List[str]] = None

"""# Sample Data as example"""

examples = [
   {"example": """Type: opinion, Opinion: Untrustworthy Source, Authors: ['Alice Johnson', 'Bob Smith'], Explanation: The source has a history of publishing unverified and often false information, Object_refs: ['obj1234', 'obj5678'], Confidence: low"""},
   {"example": """Type: opinion, Opinion: Effective Security Measure, Authors: ['Cindy Lee'], Explanation: The security measure in question has proven to effectively mitigate risks associated with cyber threats, Object_refs: ['obj91011', 'obj121314'], Confidence: high"""},
   {"example": """Type: opinion, Opinion: Outdated Technology, Authors: ['David Brown'], Explanation: The technology has not kept pace with current standards and poses security risks, Object_refs: ['obj151617'], Confidence: medium"""},
   {"example": """Type: opinion, Opinion: Reliable Software, Authors: ['Eva Green'], Explanation: The software consistently performs as advertised without significant bugs or security flaws, Object_refs: ['obj181920'], Created_by_ref: 'company123', Confidence: high"""},
   {"example": """Type: opinion, Opinion: Overvalued Asset, Authors: ['Frank Gomez', 'Heidi Klum'], Explanation: The asset does not hold as much real-world value as its market price suggests, due to inflated demand, Object_refs: ['obj212223'], Confidence: medium"""},
   {"example": """Type: opinion, Opinion: High Risk Investment, Authors: ['Ivan Petrov'], Explanation: The investment is prone to high volatility and potential losses, which outweigh the potential gains, Object_refs: ['obj242526'], Confidence: low"""},
   {"example": """Type: opinion, Opinion: Excellent Educational Content, Authors: ['Julia Chang'], Explanation: The content is well-researched, informative, and accessible, making it an excellent resource for learning, Object_refs: ['obj272829'], Confidence: high"""},
   {"example": """Type: opinion, Opinion: Poor Customer Service, Authors: ['Karl Benz'], Explanation: Numerous reports and personal experiences indicate that the customer service is unresponsive and unhelpful, Object_refs: ['obj303132'], Confidence: low"""},
   {"example": """Type: opinion, Opinion: Environmentally Friendly, Authors: ['Linda Eco'], Explanation: The company's operations and products are sustainable and minimize environmental impact, Object_refs: ['obj333435'], Created_by_ref: 'enviroGroup456', Confidence: high"""},
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
    output_schema=Opinion,
    llm=ChatOpenAI(temperature=1,model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

"""# Parameters"""

synthetic_results = synthetic_data_generator.generate(
    subject="Opinion",
    extra="Choose a unique and unconventional type for Opinion. Make the explanation realistic.",
    runs=35,
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
        'opinion': item.opinion,
        'authors': item.authors,
        'explanation': item.explanation,
        'object_refs': item.object_refs,
    })

# Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_data)

# Display the DataFrame
print(type(synthetic_df))
synthetic_df

# Save the DataFrame to a CSV file
synthetic_df.to_csv('Opinion_data.csv', index=False)  # index=False prevents adding an extra index column
print("Opinion_data data saved to 'Opinion_data.csv'")

