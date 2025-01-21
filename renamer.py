import re

import pathvalidate


from dotenv import load_dotenv
from guidance import gen, models, select, system, user, assistant
from guidance.chat import llama3_template, qwen2dot5_it_template  # noqa
import os
from pathlib import Path


load_dotenv()


MOVIES_DIR = Path(os.getenv("MOVIES_DIR"))

# model, template = (
#     "sha256-183715c435899236895da3869489cc30ac241476b4971a20285b1a462818a5b4",
#     qwen2dot5_it_template,
# )  # qwen 1.5

# model, template = (
#     "sha256-dde5aa3fc5ffc17176b5e8bdc82f587b24b2678c6c66101bf7da77af9f7ccdff",
#     llama3_template,
#     # llama 3.2-3B
# )

# model, template = (
#     "sha256-5ee4f07cdb9beadbbb293e85803c569b01bd37ed059d2715faa7bb405f31caa6",
#     qwen2dot5_it_template,
# )  # qwen2.5:3b

model, template = (
    "sha256-2bada8a7450677000f678be90653b85d364de7db25eb5ea54136ada5f3933730",
    qwen2dot5_it_template,
)  # qwen2.5:7b


path = "C:/Users/adrian/.ollama/models/blobs/" + model


def main():
    lm = models.LlamaCpp(path, echo=False, chat_template=template, n_ctx=3000)

    with system():
        lm += "Give me the clean title of the movie based on the filename.\n"
        "The title is ALWAYS in the same language as the input:\n"

    # example
    with user():
        lm += "el.gato_con.botas.(XRip, FullHD, 2012) [peliculeros.com]\n"
    with assistant():
        lm += "The input language is spanish\n"
        lm += 'The title of the movie is: "El Gato con Botas"\n'

    movies = os.listdir(MOVIES_DIR)

    for movie in movies:
        try:

            file = MOVIES_DIR / movie

            if not os.path.isfile(file):
                continue

            extension = movie.split(".")[-1]
            raw_title = movie[: -len(extension) + 1]

            with user():
                lm += raw_title + "\n"

            title = gen("title", max_tokens=50, stop='"')

            with assistant():
                lm += f'The input language is {select(["spanish", "english", "other language"])}\n'
                lm += f'The title of the movie is: "{title}"'

            new_title = lm["title"]

            year_regex = re.compile(r"(19|2[0-2])\d{2}")
            if year := year_regex.search(movie):
                new_title += f" - {year.group()}"

            new_file = pathvalidate.sanitize_filename(new_title + "." + extension)

            print(movie, " -------> ", new_file)

            file.rename(MOVIES_DIR / new_file)
        except:  # noqa
            continue


if __name__ == "__main__":
    main()
