from config import groq_client

models = groq_client.models.list()

print("\nAvailable Groq models:\n")

for model in models.data:
    print(model.id)