import bs4
import os
os.environ["QIANFAN_AK"] = ""
os.environ["QIANFAN_SK"] = ""
from langchain_community.embeddings import QianfanEmbeddingsEndpoint
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import WebBaseLoader
loader=WebBaseLoader(web_path="https://www.fzu.edu.cn/xxgk/xxjj.htm",encoding="utf-8",bs_get_text_kwargs={"separator":"\n","strip":True},bs_kwargs=dict(parse_only=bs4.SoupStrainer("div",class_="ar_article")))
document=loader.load()
from langchain_text_splitters import RecursiveCharacterTextSplitter
text_splitter = RecursiveCharacterTextSplitter(
    separators=[
        "\n\n"
        "\n",
        "。",
    ],
    chunk_size=500,
    chunk_overlap=100,
    is_separator_regex=False,
)
documents=text_splitter.split_documents(document)
db = FAISS.from_documents(documents,QianfanEmbeddingsEndpoint())
db.save_local("faiss/xxgk/xxjj")
print("数据库创建成功")
