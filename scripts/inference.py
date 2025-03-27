from transformers import pipeline

model_id = "paulbg/nebriq-model-classifier"

classifier = pipeline("text-classification", model=model_id)

text = ""

while True:
    text = input("Enter a query: ")
    
    if text == "q":
        break
    
    result = classifier(text)
    print(result)
    
print("bye!")