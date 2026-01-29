import os
import subprocess

FILES_TO_PROCESS = [
    "test1.txt","test2.txt","test3.txt","test4.txt","test5.txt","test6.txt","test7.txt","test8.txt",
    "text_categories_simple/culture.txt",
    "text_categories_simple/economy.txt",
    "text_categories_simple/politics.txt",
    "text_categories_simple/religion.txt",
    "text_categories_simple/science.txt",
    "text_categories_simple/society.txt",
    "text_categories_simple/world.txt"
]

for file in FILES_TO_PROCESS:
    if os.path.exists(file):
        os.system(f"python main.py {file}")
        print(f" Автоматический запуск {file} успешен!\n")
    else:
        print(f"Файл {file} не найден!\n")

