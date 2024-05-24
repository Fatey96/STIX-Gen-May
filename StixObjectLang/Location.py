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

"""# Schema for generating Location

"""

class Location(BaseModel):
    type: str
    spec_version: str
    id: str
    created: str
    modified: str
    Name: str
    Description: str
    latitude: float
    precision: str
    longitude: float
    region: str
    country: str
    administrative_area : str
    city: str
    street_address: str
    postal_code: str
    created_by_ref: Optional[str] = None
    revoked: Optional[str] = None
    confidence: Optional[str] = None
    lang: Optional[str] = None

"""# Sample Data as example"""

examples = [
   {"example": """Type: location, Name: Downtown Cyber Cafe, Description: A popular cyber cafe located in the heart of the city known for its high-speed internet and privacy booths., Latitude: '34.0522', Longitude: '-118.2437', Precision: 'exact', Region: 'Urban', Country: 'USA', Administrative_area: 'California', City: 'Los Angeles', Street_address: '123 Main St', Postal_code: '90015'"""},
   {"example": """Type: location, Name: Suburban Home Office, Description: A residential building in a quiet suburban neighborhood frequently used for remote work., Latitude: '51.5074', Longitude: '-0.1278', Precision: 'exact', Region: 'Suburban', Country: 'UK', Administrative_area: 'Greater London', City: 'London', Street_address: '47 Acacia Rd', Postal_code: 'SW2 3GT'"""},
   {"example": """Type: location, Name: City Park Wi-Fi Zone, Description: An open park area with public Wi-Fi access, often used by individuals seeking quiet outdoor workspaces., Latitude: '40.7128', Longitude: '-74.0060', Precision: 'approximate', Region: 'Urban Park', Country: 'USA', Administrative_area: 'New York', City: 'New York City', Street_address: 'Central Park West', Postal_code: '10023'"""},
   {"example": """Type: location, Name: Mobile Tech Van, Description: A modified van equipped with advanced communication and hacking tools, used for mobile operations., Latitude: '52.5200', Longitude: '13.4050', Precision: 'approximate', Region: 'Mobile', Country: 'Germany', Administrative_area: 'Berlin', City: 'Berlin', Street_address: 'Various Locations', Postal_code: '10117'"""},
   {"example": """Type: location, Name: Local Shopping Mall Wi-Fi Area, Description: A common area in a busy shopping mall where free Wi-Fi is available to visitors., Latitude: '48.8566', Longitude: '2.3522', Precision: 'exact', Region: 'Commercial', Country: 'France', Administrative_area: 'Ile-de-France', City: 'Paris', Street_address: '101 Mall Plaza', Postal_code: '75001'"""},
   {"example": """Type: location, Name: Riverside Bench, Description: A secluded spot by the river known for its calm environment and occasional use by remote workers., Latitude: '35.6895', Longitude: '139.6917', Precision: 'approximate', Region: 'Outdoor', Country: 'Japan', Administrative_area: 'Tokyo', City: 'Tokyo', Street_address: 'Near Kachidoki Bridge', Postal_code: '104-0061'"""},
   {"example": """Type: location, Name: Communal Library Study Room, Description: A study room in the local library, often used by students and remote workers for its quiet atmosphere., Latitude: '55.7558', Longitude: '37.6173', Precision: 'exact', Region: 'Urban', Country: 'Russia', Administrative_area: 'Moscow', City: 'Moscow', Street_address: 'Lenin Library', Postal_code: '101000'"""},
   {"example": """Type: location, Name: Underground Parking Garage, Description: A parking garage beneath a commercial complex, occasionally used for discreet meetings or operations., Latitude: '39.9042', Longitude: '116.4074', Precision: 'exact', Region: 'Subterranean', Country: 'China', Administrative_area: 'Beijing', City: 'Beijing', Street_address: 'Complex B1', Postal_code: '100044'"""},
   {"example": """Type: location, Name: Tech Startup Incubator, Description: A workspace in a tech startup incubator, known for its creative environment and high concentration of tech professionals., Latitude: '37.7749', Longitude: '-122.4194', Precision: 'exact', Region: 'Urban', Country: 'USA', Administrative_area: 'California', City: 'San Francisco', Street_address: 'Tech Hub Center', Postal_code: '94103'"""}
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
    output_schema=Location,
    llm=ChatOpenAI(temperature=1,model='gpt-4-turbo-preview'),
    prompt=prompt_template,
)

"""# Parameters"""

synthetic_results = synthetic_data_generator.generate(
    subject="Location",
    extra="Make the name, description, location data, postal code, precision, city, street_address, longtitude, latitude, region, administrative_area realistic and taken from real world geography. Never use any unrealistic place like Lost city of Atlantis. Make sure its from raal world geography (land). Avoid Landmark places like Effiel Tower, etc.",
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
        'Name': item.Name,
        'Description': item.Description,
        'latitude': item.latitude,
        'precision': item.precision,
        'longitude': item.longitude,
        'region': item.region,
        'country': item.country,
        'administrative_area': item.administrative_area,
        'street_address': item.street_address,
        'postal_code': item.postal_code,
    })

# Create a Pandas DataFrame from the list of dictionaries
synthetic_df = pd.DataFrame(synthetic_data)

# Display the DataFrame
print(type(synthetic_df))
synthetic_df

# Save the DataFrame to a CSV file
synthetic_df.to_csv('Location_data.csv', index=False)  # index=False prevents adding an extra index column
print("Location data saved to 'Location_data.csv'")

