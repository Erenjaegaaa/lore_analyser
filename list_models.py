import requests
from dotenv import dotenv_values

k = dotenv_values('.env').get('groq_api_key')
r = requests.get(
    'https://api.groq.com/openai/v1/models',
    headers={'Authorization': 'Bearer ' + k}
)
for m in sorted(r.json()['data'], key=lambda x: x['id']):
    print(m['id'])
