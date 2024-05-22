import json

documents = {
    "doc1": "Este es el primer documento. Habla sobre Python y programación.",
    "doc2": "Este segundo documento es sobre indexación de texto y búsqueda.",
    "doc3": "El tercer documento menciona Python y técnicas de búsqueda avanzada."
}

index = {}

for doc_id, text in documents.items():
    for word in text.split():
        word = word.lower().strip('.,')
        if word not in index:
            index[word] = []
        if doc_id not in index[word]:
            index[word].append(doc_id)

print("Index", index)
