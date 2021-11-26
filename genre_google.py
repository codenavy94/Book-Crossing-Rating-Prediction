import requests
from bs4 import BeautifulSoup
import json
from tqdm import tqdm
import time
from random import randint

#!pip install lxml

max_time = 5

#isbns1 <- 이거만 숫자 바꾸기
f=open('isbns2.txt','r')
isbns1=f.readlines()
f.close()

genre_dict={}
for isbn in tqdm(isbns1):
    url='https://www.google.com/search?q='
    base_url=url+'ISBN+'+isbn
    #print(base_url)
    
    headers = {'user-agent':'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/96.0.4664.45 Safari/537.36'}
    res = requests.get(base_url, headers=headers)
    wait_time = randint(1, max_time)
    time.sleep(wait_time)

    soup = BeautifulSoup(res.text, 'lxml')
    try:
        book_info = soup.find_all('span', attrs={'class':"w8qArf"})
        for info in book_info:
            if '장르' in info.text:
                genre=info.next_sibling.text
                # print(isbn, genre)
                genre_dict[f'{isbn}']=genre
                print('INFO ADDED TO DICT')
            else:
                continue
    except:
        pass
    
# print(genre_dict)

with open('genre2.json','w') as fp:
    json.dump(genre_dict, fp)