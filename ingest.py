import json
from PyPDF2 import PdfReader


def extract_pages_from_pdf(pdf_path):
    docs = []
    metadatas = []
    with open(pdf_path, 'rb') as pdf_file:
        reader = PdfReader(pdf_file)
        for page in range(15,458):
            text = reader.pages[page].extract_text()
            docs.append(text)
            metadatas.append({"source": "Page: " + str(page-5)})
    return docs, metadatas


if __name__ == '__main__':
    pdf_path = 'progit.pdf'
    docs, metadatas = extract_pages_from_pdf(pdf_path)

    with open("docs.json", "w") as f:
        json.dump(docs, f)

    with open("metadatas.json", "w") as f:
        json.dump(metadatas, f)
