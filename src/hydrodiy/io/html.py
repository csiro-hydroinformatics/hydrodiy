from datetime import datetime
import pandas as pd
from pathlib import Path

FHERE = Path(__file__).resolve().parent

HTML_TEMPLATE = '''
<!DOCTYPE html>
<html>

<head>

<style>
{css_style}
</style>

<title>{title}</title>

</head>

<body>
<h1>{title}</h1>
<p><b>Author</b>: {author}</p>
<p><b>Date</b>: {now}</p>
<p><b>Comment</b>: {comment}</p>
<hr>
{table}
</body>
</html>.
'''

def dataframe2html(df, fhtml, title, comment="", author="", index=False):
    # Open CSS
    fcss = FHERE / "pandas_dataframe_style.css"
    with fcss.open("r") as fo:
        css_style = fo.read()

    # Format table
    with fhtml.open("w") as fo:
        fo.write(HTML_TEMPLATE.format(\
            author=author, \
            css_style=css_style, \
            now=str(datetime.now()), \
            comment=comment, \
            table=df.to_html(index=index, classes="css_style"),\
            title=title
        ))


