
import PyPDF2
import re
import nltk
# nltk.download('punkt')
import spacy
from textblob import TextBlob
from transformers import pipeline
summarizer = pipeline("summarization", model="Junlaii/bart_4acticle_abstract")

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-zh")
model = AutoModelForSeq2SeqLM.from_pretrained("Helsinki-NLP/opus-mt-en-zh")

def translate_text_to_chinese(text):
    """
    The function `translate_text_to_chinese` takes in a text as input and uses a tokenizer and a model to translate the text
    into Chinese.

    :param text: The `text` parameter is the input text that you want to translate to Chinese
    :return: the translated text in Chinese.
    """
    input_ids = tokenizer.encode(text, return_tensors="pt")
    translated_ids = model.generate(input_ids)
    translated_text = tokenizer.decode(translated_ids[0], skip_special_tokens=True)
    return translated_text


def segment_text_blob(text):
    """
    The function `segment_text_blob` takes a text as input and returns a list of paragraphs, where each paragraph is a
    string consisting of one or more sentences.

    :param text: The `text` parameter is a string that represents the text that needs to be segmented into paragraphs
    :return: a list of paragraphs, where each paragraph is a string containing one or more sentences.
    """
    blob = TextBlob(text)
    sentences = blob.sentences
    paragraphs = []
    current_paragraph = []

    for sentence in sentences:
        current_paragraph.append(sentence)

        if str(sentence).endswith('.'):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    return paragraphs

def segment_text_nltk(text):
    """
    The function `segment_text_nltk` takes in a text and uses the NLTK library to segment it into paragraphs based on
    periods.

    :param text: The `text` parameter is a string that represents the input text that you want to segment into paragraphs
    :return: The function `segment_text_nltk` returns a list of paragraphs. Each paragraph is a string that contains a
    sequence of sentences.
    """
    nlp = spacy.load('en_core_web_sm')
    doc = nlp(text)
    paragraphs = []
    current_paragraph = []

    for token in doc:
        current_paragraph.append(token.text)

        if token.text.endswith('.'):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    return paragraphs

def segment_text_spacy(text):
    """
    The function `segment_text_spacy` takes in a text and segments it into paragraphs based on the presence of periods at
    the end of sentences.

    :param text: The `text` parameter is a string that represents the input text that you want to segment into paragraphs
    :return: a list of paragraphs, where each paragraph is a string containing one or more sentences.
    """
    sentences = nltk.sent_tokenize(text)
    paragraphs = []
    current_paragraph = []

    for sentence in sentences:
        current_paragraph.append(sentence)

        if sentence.endswith('.'):
            paragraphs.append(' '.join(current_paragraph))
            current_paragraph = []

    return paragraphs



def split_paragraphs(text):
    """
    The function `split_paragraphs` takes a text input and splits it into paragraphs based on empty lines.

    :param text: The `text` parameter is a string that represents a paragraph or multiple paragraphs of text
    :return: a list of paragraphs split from the input text.
    """
    paragraphs = re.split(r'\n\s*\n', text)
    return paragraphs

def extract_text_from_pdf(file_path):
    """
    The function `extract_text_from_pdf` takes a file path as input, reads a PDF file at that path, and returns the
    extracted text from all pages of the PDF.

    :param file_path: The file path is the location of the PDF file that you want to extract text from. It should be a
    string that specifies the path to the file on your computer. For example, "C:/Documents/example.pdf" or
    "/home/user/Documents/example.pdf"
    :return: the extracted text from the PDF file.
    """
    with open(file_path, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        return text

def print_graph(result):
    """
    The `print_graph` function is used to print the paragraphs in the `result` list, with each paragraph numbered.

    :param result: The `result` parameter is a list of strings. Each string represents a paragraph of text
    """
    # 打印结果
    for i, paragraph in enumerate(result):
        print(f"段落 {i + 1}: {paragraph.strip()}")

def get_reps(ARTICLE):
    """
    The function "get_reps" takes an article as input and returns a summarized version of the article.

    :param ARTICLE: The ARTICLE parameter is the text of the article that you want to summarize
    :return: The function `get_reps` returns the summary text of the given article.
    """
    ret = summarizer(ARTICLE, max_length=512, min_length=3, do_sample=False)[0]['summary_text']
    return ret



def main2(t_from_pdf,dilogs=10):
    """
    The main2 function takes a text input, segments it into smaller chunks based on the specified number of dialogs, and
    then performs some operations on each chunk before returning the results.

    :param t_from_pdf: The `t_from_pdf` parameter is the text extracted from a PDF file. It is the input text that needs to
    be processed
    :param dilogs: The parameter "dilogs" is used to specify the number of segments or chunks you want to divide the text
    into. It determines how many segments the text will be divided into for processing. The default value is 10, meaning the
    text will be divided into 10 segments if no value is, defaults to 10 (optional)
    :return: The function `main2` returns a list of translated texts.
    """
    text = segment_text_spacy(t_from_pdf)
    solve_list = []
    tmp = ''
    for i in range(len(text)):
        if i % int(dilogs) == 0:
            solve_list.append(tmp)
            tmp = ''
        tmp = tmp + text[i]
    if tmp != '':
        solve_list.append(tmp)
    ret = []
    for i in solve_list:
        tmp = get_reps(i)
        print('tmp: ', tmp)
        ret.append(translate_text_to_chinese(tmp))
    return  ret


