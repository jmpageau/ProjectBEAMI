import spacy
from spacy.tokens import DocBin
import json

def convert_data(input_file, output_file):
    nlp = spacy.blank("fr")
    db = DocBin()

    with open(input_file, "r", encoding="utf-8") as f:
        data = json.load(f)

    for text, annotations in data:
        doc = nlp.make_doc(text)
        ents = []
        for start, end, label in annotations["entities"]:
            span = doc.char_span(start, end, label=label)
            if span is None:
                print(f"Skipping entity [{start}, {end}, {label}] in text: {text}")
            else:
                ents.append(span)
        doc.ents = ents
        db.add(doc)

    db.to_disk(output_file)

# Convert train and dev datasets
convert_data("annotations_test.json", "train.spacy")
convert_data("annotations_test.json", "dev.spacy")
