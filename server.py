import gradio as gr
import PyPDF2
import logging
from main import main2

# 标题
from transformers import pipeline

title = "easy article"
# 标题下的描述，支持md格式
description = "自动提取和总结文章大意"
# 页面最后的信息，可以选择引用文章，支持md格式
article = "Easy to use"

# 配置日志输出的格式
logging.basicConfig(
    level=logging.INFO,  # 设置日志级别为 INFO，可根据需要调整
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

def read_pdf(file):
    """
    The function `read_pdf` reads the text content from a PDF file and returns it as a string.

    :param file: The "file" parameter is the PDF file that you want to read. It should be a file object that has been opened
    in binary mode ('rb')
    :return: the extracted text from the PDF file.
    """
    # 打开PDF文件
    with open(file.name, 'rb') as file:
        pdf_reader = PyPDF2.PdfReader(file)
        num_pages = len(pdf_reader.pages)

        text = ""
        for page_num in range(num_pages):
            page = pdf_reader.pages[page_num]
            text += page.extract_text()

        return text


def transformers(tmp,dilogs):
    """
    The function "transformers" calls another function "main2" with two arguments "tmp" and "dilogs" and returns the result.

    :param tmp: The "tmp" parameter is likely a temporary variable or placeholder that is being passed into the
    "transformers" function. Without more context, it is difficult to determine the exact purpose or meaning of this
    variable
    :param dilogs: The parameter "dilogs" seems to be misspelled. Did you mean "dialogs"?
    :return: The function `transformers` is returning the result of calling the function `main2` with the arguments `tmp`
    and `dilogs`.
    """
    return main2(tmp,dilogs)


def main(dilogs,file):
    """
    The main function reads a PDF file, applies a transformation to it based on the number of dilogs specified, and returns
    the transformed result.

    :param dilogs: The "dilogs" parameter is an integer that represents the number of dialogues you want to extract from the
    PDF file
    :param file: The "file" parameter is the path to the PDF file that you want to read and process
    :return: the variable "tmp".
    """
    return transformers(read_pdf(file),int(dilogs))



if __name__ == '__main__':
    iface = gr.Interface(fn=main, inputs=[
        gr.inputs.Textbox(label="段落数"),
        "file"], outputs="text")
    print('please open https://127.0.0.1:8080')
    iface.launch(server_name="127.0.0.1", server_port=8080, share=True)
