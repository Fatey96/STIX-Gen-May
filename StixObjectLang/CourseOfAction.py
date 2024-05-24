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

"""# Schema for generating CourseOfAction"""

class CourseOfAction(BaseModel):
    type: str
    spec_version: str
    id: str
    created: str
    modified: str
    created_by_ref: Optional[str] = None
    revoked: Optional[str] = None
    confidence: Optional[str] = None

"""# Sample Data as example"""

examples = [
{"example": """Type: course-of-action, Name: Implement Multi-Factor Authentication, Description: This action mandates the adoption of multi-factor authentication for all user accounts to significantly decrease the likelihood of unauthorized access. It involves using two or more verification methods for user authentication, adding an extra layer of security beyond just passwords."""},
{"example": """Type: course-of-action, Name: Regular Software Patching, Description: A systematic approach to regularly update and patch operating systems and software applications. This course of action aims to fix vulnerabilities that could be exploited by attackers, keeping the organization's digital assets secure."""},
{"example": """Type: course-of-action, Name: Cybersecurity Awareness Training, Description: An educational initiative aimed at all organizational members to heighten awareness about cyber threats such as phishing, social engineering, and how to prevent them. This course of action promotes a culture of security and vigilance."""},
{"example": """Type: course-of-action, Name: Data Encryption Strategy, Description: A security measure that involves encrypting sensitive data, both at rest and in transit. This course of action ensures that even in the event of a data breach, the information remains secure and inaccessible to unauthorized parties."""},
{"example": """Type: course-of-action, Name: Incident Response Plan, Description: The establishment of a formalized plan to respond to cybersecurity incidents. This includes identifying, containing, eradicating, and recovering from incidents to minimize impact and prevent future occurrences."""},
{"example": """Type: course-of-action, Name: Secure Configuration Standards, Description: The development and enforcement of secure configurations for all IT systems and applications to reduce vulnerabilities and safeguard against attacks. This includes disabling unnecessary services and applying the principle of least privilege."""},
{"example": """Type: course-of-action, Name: Network Segmentation, Description: Dividing the network into distinct zones to improve security and control. This course of action limits the spread of breaches by segregating sensitive areas and applying strict access controls."""},
{"example": """Type: course-of-action, Name: Continuous Monitoring, Description: The implementation of tools and practices for the ongoing surveillance of IT systems to detect and respond to threats in real time. This course of action helps in identifying suspicious activities and mitigating threats before they can cause significant damage."""},
{"example": """Type: course-of-action, Name: Disaster Recovery Planning, Description: The process of creating a comprehensive plan for the continuation or recovery of systems in the event of a catastrophic failure. This course of action includes strategies for data backup, system restoration, and maintaining business operations under adverse conditions."""}
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
    output_schema=CourseOfAction,
    llm=ChatOpenAI(temperature=1,model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

"""# Parameters"""

synthetic_results = synthetic_data_generator.generate(
    subject="course_of_action",
    extra="Choose a unique and unconventional type for each course of action.",
    runs=75,
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
    })

# Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_data)

# Display the DataFrame
print(type(synthetic_df))
synthetic_df

# Save the DataFrame to a CSV file
synthetic_df.to_csv('course_of_action.csv', index=False)  # index=False prevents adding an extra index column
print("course_of_action data saved to 'course_of_action.csv'")

