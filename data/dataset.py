import pandas as pd
import random

simple_queries = [
    "Hello there", "What’s the weather like", "How tall am I", "Good morning",
    "What day is it", "Turn on the lights", "Play some music", "How old are you",
    "What’s for dinner", "Is it raining", "Tell me a joke", "Where’s my phone",
    "What’s 2 plus 2", "How’s your day", "Do you like coffee", "What’s my name",
    "Send a text", "Are we there yet", "What’s the score", "How do I boil water",
    "Pick a color", "What time is lunch", "Is the store open", "How far is it",
    "Say something nice", "What’s on TV", "Can I have a cookie", "Where’s the dog",
    "How do I spell cat", "What’s the temperature", "Are you busy", "Call mom",
    "What’s the capital of France", "How many days in a week", "Is it dark outside",
    "What’s the smell like", "Do you know me", "How loud is it", "What’s 5 minus 3",
    "Where’s the bathroom", "Can you sing", "How’s the food", "What’s a dog",
    "Is this fun", "What color is the sky", "How do I wave", "Are you awake",
    "What’s my age", "How big is a cat", "What’s the time now", "Do you see me",
    "What’s a tree", "How hot is it", "Can I sleep now", "Where’s my hat",
    "What’s 10 times 2", "Is it cold", "How do I sit", "What’s your name",
    "Can you dance", "What’s a bird", "How do I eat", "Are we friends",
    "What’s the moon", "How high is that", "Is this loud", "What’s a car"
]

medium_queries = [
    "How do I bake a cake", "What’s the best phone to buy", "Can you explain gravity",
    "Plan a trip to Paris", "How does a car engine work", "What’s a good book to read",
    "Teach me basic Spanish", "How do I fix a leaky faucet", "What’s climate change",
    "Design a simple website", "How does Wi-Fi work", "What’s the plot of Hamlet",
    "Make a budget for me", "How do I grow tomatoes", "Explain how vaccines work",
    "What’s the history of jazz", "How do I paint a room", "Suggest a gift for mom",
    "What’s the rules of soccer", "How do I train a puppy", "Explain the water cycle",
    "What’s a balanced diet", "How do I sew a button", "Plan a picnic day",
    "What’s photosynthesis", "How does recycling work", "Teach me to draw a face",
    "What’s the meaning of life", "How do I invest money", "Explain the solar system",
    "What’s a good workout", "How do I write a poem", "What’s the history of tea",
    "How do I tie a tie", "Explain how planes fly", "What’s impressionism art",
    "How do I make coffee", "Plan a date night", "What’s the science of rainbows",
    "How do I clean a laptop", "Teach me about dinosaurs", "What’s a mortgage",
    "How do I meditate", "Explain electricity basics", "What’s the story of Rome",
    "How do I play chess", "Suggest a hobby for kids", "What’s the life of a bee",
    "How do I make soap", "Explain how clocks work", "What’s a good movie plot",
    "How do I knit a scarf", "Teach me about stars", "What’s the history of bread",
    "How do I build a shelf", "Explain ocean tides", "What’s a healthy breakfast",
    "How do I take photos", "Plan a family game night", "What’s the role of bees",
    "How do I fix a bike", "Explain how sound works", "What’s the culture of Japan",
    "How do I start a blog", "Teach me about wine", "What’s the history of dance"
]

advanced_queries = [
    "Analyze quantum mechanics", "Design a sustainable city", "Explain neural networks",
    "What caused the Big Bang", "Debate ethics of AI", "Model climate change effects",
    "Break down relativity theory", "How do black holes form", "Predict economic trends",
    "Analyze Shakespeare’s themes", "Develop a machine learning app", "Explain DNA sequencing",
    "What’s consciousness science", "Design a space mission", "Debate free will",
    "Model a pandemic spread", "Explain string theory", "Analyze ancient philosophy",
    "Develop a renewable energy plan", "What’s the psychology of love", "Simulate a stock market",
    "Explain dark matter", "Analyze global trade impacts", "Design an AI ethics code",
    "What’s the origin of language", "Model planetary orbits", "Debate genetic engineering",
    "Explain blockchain tech", "Analyze Renaissance art", "Design a self-driving car system",
    "What’s the physics of time", "Predict AI’s future role", "Explain particle physics",
    "Analyze cultural evolution", "Design a water purification system", "Debate space colonization",
    "Model quantum computing", "Explain the multiverse", "Analyze historical revolutions",
    "Develop a cryptography system", "What’s the biology of aging", "Simulate galaxy formation",
    "Explain chaos theory", "Analyze postmodern literature", "Design a robotic arm",
    "What’s the chemistry of stars", "Predict geopolitical shifts", "Explain game theory",
    "Analyze mental health trends", "Design a fusion reactor", "Debate universal basic income",
    "Model ecological systems", "Explain cosmic inflation", "Analyze existentialism",
    "Develop a virtual reality world", "What’s the sociology of power", "Simulate brain functions",
    "Explain gravitational waves", "Analyze modern architecture", "Design a telemedicine platform",
    "What’s the anthropology of religion", "Predict tech innovation", "Explain superconductivity",
    "Analyze war strategy history", "Design an exoplanet explorer", "Debate quantum ethics"
]

# Combine into dataset
data = (
    [(text, "simple") for text in simple_queries] +
    [(text, "medium") for text in medium_queries] +
    [(text, "advanced") for text in advanced_queries]
)

random.shuffle(data)

df = pd.DataFrame(data, columns=["text", "label"])

df.to_csv("dataset.csv", index=False)
print("Dataset saved as 'dataset.csv' with 200 examples.")