import os
from typing import List, Optional
import dotenv
from langchain_core.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field, validator
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

# Define the Location schema
class Location(BaseModel):
    type: str = Field(default="location")
    spec_version: str = Field(default="2.1")
    id: str = Field(description="Unique identifier for the location")
    created: str = Field(description="Creation date of the location entry")
    modified: str = Field(description="Last modification date of the location entry")
    name: Optional[str] = Field(default=None, description="A name used to identify the Location")
    description: Optional[str] = Field(default=None, description="A textual description of the Location")
    latitude: Optional[float] = Field(default=None, description="The latitude of the Location in decimal degrees")
    longitude: Optional[float] = Field(default=None, description="The longitude of the Location in decimal degrees")
    precision: Optional[float] = Field(default=None, description="Defines the precision of the coordinates in meters")
    region: Optional[str] = Field(default=None, description="The region that this Location describes")
    country: Optional[str] = Field(default=None, description="The country that this Location describes (ISO 3166-1 ALPHA-2 Code)")
    administrative_area: Optional[str] = Field(default=None, description="The state, province, or other sub-national administrative area")
    city: Optional[str] = Field(default=None, description="The city that this Location describes")
    street_address: Optional[str] = Field(default=None, description="The street address that this Location describes")
    postal_code: Optional[str] = Field(default=None, description="The postal code for this Location")
    @validator('latitude')
    def validate_latitude(cls, v):
        if v is not None and (v < -90.0 or v > 90.0):
            raise ValueError('Latitude must be between -90.0 and 90.0')
        return v

    @validator('longitude')
    def validate_longitude(cls, v):
        if v is not None and (v < -180.0 or v > 180.0):
            raise ValueError('Longitude must be between -180.0 and 180.0')
        return v

    @validator('precision')
    def validate_precision(cls, v, values):
        if v is not None and ('latitude' not in values or 'longitude' not in values):
            raise ValueError('Latitude and longitude must be present if precision is specified')
        return v

# Sample data as examples
examples = [
    {"example": """Type: location, Spec Version: 2.1, ID: loc-002, Created: 2024-07-10T09:15:00Z, Modified: 2024-07-10T09:15:00Z, Name: Chiang Mai Old City, Description: Historic center of Chiang Mai, Thailand, Latitude: 18.7883, Longitude: 98.9853, Precision: 100, Region: south-eastern-asia, Country: th, Administrative Area: Chiang Mai Province, City: Chiang Mai, Postal Code: 50200"""},
    {"example": """Type: location, Spec Version: 2.1, ID: loc-003, Created: 2024-07-10T09:30:00Z, Modified: 2024-07-10T09:30:00Z, Name: Eiffel Tower, Description: Iconic iron lattice tower on the Champ de Mars in Paris, Latitude: 48.8584, Longitude: 2.2945, Precision: 5, Region: western-europe, Country: fr, Administrative Area: Île-de-France, City: Paris, Street Address: Champ de Mars, 5 Avenue Anatole France, Postal Code: 75007"""}
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
    output_schema=Location,
    llm=ChatOpenAI(temperature=1, model='gpt-4o'),
    prompt=prompt_template,
)

def generate_locations(count: int) -> List[Location]:
    """
    Generate synthetic location entries.
    
    Args:
        count (int): Number of location entries to generate.
    
    Returns:
        List[Location]: List of generated location entries.
    """
    synthetic_results = synthetic_data_generator.generate(
        subject="location",
        extra="Create diverse location entries with a mix of regional, country-level, and precise coordinate-based locations. No landmarks. Ensure realistic values for all fields and adherence to specified formats (e.g., ISO 3166-1 ALPHA-2 for country codes).",
        runs=count,
    )
    return synthetic_results