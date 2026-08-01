"""
pnr_generator_bedrock.py
Test mode: loads flights from data/flights.json (no API call)
then generates synthetic PNR database using AWS Bedrock.
"""
import boto3
import json
import re
import os
from dotenv import load_dotenv

load_dotenv()

# AWS Bedrock client
bedrock = boto3.client(
    service_name='bedrock-runtime',
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

MODEL_ID = "us.anthropic.claude-haiku-4-5-20251001-v1:0"

def get_flights():
    if os.path.exists('data/flights.json'):
        print("Loading saved flights - no API call!")
        with open('data/flights.json', 'r') as f:
            return json.load(f)
    else:
        print("No saved flights found - run flight_fetcher.py first!")
        return []

def generate_pnr_database(flights):
    flight_list = "\n".join([
        f"{f['flight_iata']} from {f['origin']} to {f['destination']}"
        for f in flights
    ])

    print("Sending these flights to Claude (Bedrock):")
    print(flight_list)

    prompt = f"""
    Create a realistic passenger PNR database for these exact flights only:
    {flight_list}

    Generate 10 passengers using ONLY the flight codes listed above.
    For each passenger, assign a realistic cabin_class ("economy", "premium_economy", or "business")
    and a fare_amount in USD consistent with that class (economy: 150-600, premium_economy: 400-1200,
    business: 1500-6000).

    Use exactly this format:
    {{
        "PNR_CODE": {{
            "passenger_name": "Full Name",
            "flight_iata": "flight code from the list above",
            "destination": "airport code",
            "contact_email": "email@example.com",
            "cabin_class": "economy|premium_economy|business",
            "fare_amount": 000
        }}
    }}
    """

    response = bedrock.converse(
        modelId=MODEL_ID,
        system=[{"text": "You are a JSON generator. Return pure valid JSON only. No markdown. No code fences. No explanation."}],
        messages=[{"role": "user", "content": [{"text": prompt}]}],
        inferenceConfig={"maxTokens": 1000}
    )

    raw = response['output']['message']['content'][0]['text']
    cleaned = re.sub(r'```json\n?|```', '', raw).strip()
    return json.loads(cleaned)

if __name__ == "__main__":
    flights = get_flights()
    pnr_dict = generate_pnr_database(flights)

    with open('data/pnr_database.json', 'w') as f:
        json.dump(pnr_dict, f, indent=2)

    print("PNR database saved!")
    print(pnr_dict)
