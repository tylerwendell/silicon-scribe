import os
import click
import pandas as pd
import logging
import asyncio
from concurrent.futures import ThreadPoolExecutor
from queue import Queue
import typing
from typing import Any
from typing import TextIO
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from language_enum import LanguageEnum

original_language = LanguageEnum.GREEK

def setup_logging():
    """
    Configure the logger.
    """
    logging.basicConfig(level=logging.INFO)
    log_formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

    # Create a log file handler
    log_file_handler = logging.FileHandler('translation.log')
    log_file_handler.setFormatter(log_formatter)

    # Create a console (stdout) handler
    log_console_handler = logging.StreamHandler()
    log_console_handler.setFormatter(log_formatter)

    # Get the root logger and add handlers
    root_logger = logging.getLogger()
    root_logger.addHandler(log_file_handler)
    root_logger.addHandler(log_console_handler)

def file_writer(filename: str)->TextIO:
    """
    Open a file for writing, creating any necessary directories.
    """
    logging.info("Opening File %s to Write", filename)
    folder_path = os.path.dirname(filename)
    os.makedirs(folder_path, exist_ok=True)
    try:
        return open(filename, 'w')
    except IOError as e:
        logging.error(f"Failed to open file {filename}: {e}")
        raise

def open_source(source_book: str, delimiter: str)->pd.DataFrame:
    """
    Open the source book.
    """
    logging.info("Loading Book of {}...".format(source_book))
    try:
        source_data = pd.read_csv('data/scripture/{}'.format(source_book), sep=delimiter)
        logging.info("Done!")
        return source_data.dropna(subset=['Text'])
    except Exception as e:
        logging.error(f"Failed to load {source_book}: {e}")
        raise

def text_batch(text_list: list, index: int, batch_size: int) -> str:
    """
    Create a batch of text from a list.
    """
    ceiling = min(len(text_list), index + batch_size)
    return ' '.join(text_list[index:ceiling])

async def translate(tokenizer, model, text, language)->str:
    """
    Translate text to a target language.
    """
    # print(tokenizer)
    # print(model)
    # print(text)
    inputs = tokenizer(text, return_tensors="pt")
    translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.lang_code_to_id[language.value], max_new_tokens=500)
    return tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]

def write_header(file: TextIO, tokenizer: Any, model: Any, language: LanguageEnum, text:str, f):
    if language != LanguageEnum.ENGLISH:
        text = translate(tokenizer, model, text, language.value)
    file.write(f.format(text))

def concurrently_translate(file: TextIO, tokenizers: list, models: list, language: LanguageEnum, queue: Queue, concurrency: int):
    while not queue.empty():
        tasks=[]
        for i in range(concurrency):
            if not queue.empty():
                t=tokenizers[i]
                m=models[i]
                text = queue.get()
                l=language
                tasks.append(translate(t,m,text, l))
            else:
                break
        loop = asyncio.get_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(asyncio.gather(*tasks))
        file.write(' '.join(result))

def tokenizers(num:int)->list:
    tokenizer_list=[]
    for i in range(num):
        tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=original_language.value)
        tokenizer_list.append(tokenizer)
    return tokenizer_list

def models(num:int)->list:
    model_list=[]
    for i in range(num):
        model = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
        model_list.append(model)
    return model_list

@click.command()
@click.argument('languages', nargs=-1, type=LanguageEnum)
@click.option('-c', '--concurrency', type=int, default=1)
@click.option('-b', '--batch', type=int, default=3)
def main(languages, concurrency, batch):
    """
    Translate a text to multiple languages and save the results.
    """
    book = "The Gospel of John"
    book_short = "John"
    source=open_source("test.txt","|")
    queue = Queue()

    logging.info("Loading Tokenizers and Model...")
    # tokenizer = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=original_language.value)
    ts = tokenizers(concurrency)
    tokenizer_eng = AutoTokenizer.from_pretrained("facebook/nllb-200-3.3B", src_lang=LanguageEnum.ENGLISH.value)
    model_eng = AutoModelForSeq2SeqLM.from_pretrained("facebook/nllb-200-3.3B")
    # print("ENG tokenizer: ", tokenizer_eng)
    # print("ENG model", model_eng)
    ms = models(concurrency)
    logging.info("Done!")

    for lang in languages:
        logging.info("Translating for language: %s", lang.value)
        f = file_writer("data/translated_books/{}/{}.md".format(lang.value, book_short))
        write_header(f, tokenizer_eng, model_eng, lang, book_short, "\n\n# {}\n\n" )
        f.write("# {}".format(book_short))
        for chapters in source["Chapter"].unique():
            chapter_name="Chapter {}".format(chapters)
            logging.info(chapter_name)
            write_header(f, tokenizer_eng, model_eng, lang, chapter_name, "\n\n## {}\n\n")
            section = source.loc[source['Chapter'] == chapters]
            text_list = section["Text"].tolist()
            for i in range(0, len(text_list), batch):
                queue.put(text_batch(text_list, i, batch))
                # translated = translate(ts, ms, to_translate, lang.value)
                # f.write(translated+ " ")
            concurrently_translate(f, ts, ms,lang, queue, concurrency)
        f.close()



if __name__ == "__main__":
    setup_logging()
    main()
