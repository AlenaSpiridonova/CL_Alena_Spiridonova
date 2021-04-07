import requests as rq
from bs4 import BeautifulSoup
from pycorenlp import StanfordCoreNLP
import json
import re

nlp_wrapper = StanfordCoreNLP('http://localhost:9000')
# java -mx4g -cp "*" edu.stanford.nlp.pipeline.StanfordCoreNLPServer -timeout 10000000 -annotators tokenize
# -tokenize.options "splitHyphenated=false" # слова с дефисом пока не разделяются на отдельные составляющие


# Функция для получения слов на перевод
def get_words_to_be_translated(text, unwanted_words):
    # обращаемся к серверу Stanford для обработки текста, а именно для определения частей речи и лемм слов и парсинга
    # зависимостей
    server_response = nlp_wrapper.annotate(text, properties={
        'annotators': 'pos, lemma, depparse',
        'outputFormat': 'JSON',
        'timeout': 100000000000,
    })
    server_response = json.loads(server_response)  # получаем от сервера ответ в формате json
    # создаем список частей речи, которые не нужны для словаря (предлоги, частицы, имена собственные и т. д.)
    pos_deletion_list = ['PRP', 'DT', 'IN', 'TO', 'NNP', 'CC', 'CD', 'EX', 'JJR', 'JJS', 'LS', 'MD', 'PDT', 'RP',
                         'PRP$', 'RBR', 'RBS', 'UH', 'WDT', 'WP', 'WP$', 'WRB', 'NNPS', 'POS', 'SYM']
    words_list = []
    # обрабатываем результат в формате json
    for sentence in server_response["sentences"]:
        for word in sentence["tokens"]:
            if word["pos"] not in pos_deletion_list and word["lemma"] not in unwanted_words and len(word["lemma"]) > 1 \
                    and word["lemma"] not in words_list:
                words_list.append(word["lemma"])
            for dep in sentence["basicDependencies"]:  # извлекаем фразовые глаголы через парсинг зависимостей
                if dep['dep'] == "compound:prt" and word["word"] == dep["governorGloss"]:
                    dep["governorGloss"] = word["lemma"]
                    phrasal_verb = dep["governorGloss"] + ' ' + dep["dependentGloss"]
                    if phrasal_verb not in words_list:
                        words_list.append(phrasal_verb)
    unhyphenated_words_list = []
    # перебираем слова из words_list для обработки слов, написанных через дефис
    for word in words_list:
        if '-' not in word:
            unhyphenated_words_list.append(word)
        else:
            if re.search(r'\w-\w-', word):  # не берем в словарь слова, растягиваемые говорящим, типа
                # he-e-e-e-e-e-e-elp или n-n-n-nothing
                continue
            # другие слова с дефисом берем как целиком, так и по частям (если они не относятся к исключаемым словам)
            else:
                split_list = (re.sub(r'-', ' ', word)).split()
                word_parts_list = []
                [word_parts_list.append(word_part) for word_part in split_list if word_part not in word_parts_list and
                 word_part not in unhyphenated_words_list and len(word_part) > 1]
                unhyphenated_words_list.extend(word_parts_list)
                if not word.startswith('un-'):  # избавляемся от слов с приставкой un-, т. к. основа слова будет \
                    # переведена в любом случае
                    unhyphenated_words_list.append(word)
    final_list = []
    for word in unhyphenated_words_list:
        if word in unwanted_words or re.search(r'\d+', word) or not re.search(r'[a-zA-Z]', word):  # среди полученных \
            # слов попадаются цифры либо слова, содержащие цифры, а также китайские символы, поэтому они сразу удаляются
            # за ненадобностью
            continue
        else:
            final_list.append(word)
    words_list = final_list
    return words_list


# Функция для извлечения перевода слов
def get_translations(words_list):
    translated_dict = {}  # результатом выполнения функции является словарь вида {слово: перевод слова}
    for word_to_translate in words_list:  # итерируемся по всем словам из списка
        try:
            main_url = 'https://wooordhunt.ru/word/'  # сохраняем в переменную ссылку на страницу со словарем En-Ru
            word_url = main_url + word_to_translate  # формируем url страницы с переводом из двух частей
            response = rq.get(word_url)  # сохраняем ответ сервера на запрос к странице с текстом
            response.encoding = 'utf-8'
            response_html = response.text  # сохраняем исходный код страницы из ответа сервера
            response_soup = BeautifulSoup(response_html, 'html.parser')  # делаем из него суп
            transcription_div = response_soup.find('div', class_='trans_sound')  # находим нужный div
            transcription_span = transcription_div.find('span')  # затем в нем - транскрипцию слова
            translation = response_soup.find('div', class_='t_inline_en')  # затем перевод
            if response_soup.find('div', id='word_forms'):  # проверяем, имеются ли у слова, в частности, глагола,
                # другие формы
                word_forms = response_soup.find('div', id='word_forms')
                forms = re.findall(r'(?<=</span> )\S*(?=\s*<)', str(word_forms))
                if len(forms) > 1:  # если они есть, то нужных форм должно быть две (а не одна, как, напр., woman у
                    # women)
                    irreg_forms = '(' + ', '.join(forms) + ')'  # записываем формы вместе с переводом слова в словарь
                    translation_result = ' ' + irreg_forms + transcription_span.text + ' – ' + translation.text
                    translated_dict[word_to_translate] = translation_result
                else:
                    translation_result = transcription_span.text + ' – ' + translation.text  # а если двух форм нет,
                    translated_dict[word_to_translate] = translation_result  # записываем перевод без них
            else:
                translation_result = transcription_span.text + ' – ' + translation.text
                translated_dict[word_to_translate] = translation_result
        except Exception:
            try:
                main_url = 'https://www.multitran.com/m.exe?l1=1&l2=2&s='  # сохраняем в переменную ссылку на страницу с
                # другим словарем En-Ru
                word_to_find = re.sub(r' ', '+', re.sub(r'’', '%27', word_to_translate))  # слово/выражение приводится
                # в нужный для добавления к ссылке вид
                word_url = main_url + word_to_find  # формируем url страницы из двух частей
                response = rq.get(word_url)  # сохраняем ответ сервера на запрос к странице с текстом
                response.encoding = 'utf-8'
                response_html = response.text  # сохраняем исходный код страницы из ответа сервера
                response_soup = BeautifulSoup(response_html, 'html.parser')  # делаем из него суп
                transcription_div = (response_soup.find_all('div', class_='middle_col')[2])  # находим нужный div
                transcription_tr = (transcription_div.find_all('tr')[1])  # находим tr с транскрипцией
                transcription_span = transcription_tr.find('span')  # находим span с транскрипцией
                transcription_a = transcription_tr.find('a')  # находим a, где записано переводимое слово
                transcription_a_text = transcription_a.text.lower()  # приводим запись к нижнему регистру, чтобы она
                # совпадала с искомым словом
                if transcription_a_text != word_to_translate:  # если перевод слова находится только частично, напр.,
                    # только semi- в слове semi-rational, то его перевод не принимается
                    continue
                if transcription_span:  # если транскрипция нашлась, то она записывается в переменную
                    transcription_span_text = ' ' + re.sub(r"\'", "ˈ", transcription_span.text)  # заменяем апостроф,
                    # служащий ударением в транскрипции в "Мультитране", на другой знак, который не будет мешать
                    # считыванию словаря в дальнейшем
                else:
                    transcription_span_text = ''
                translation_output = []  # создается список, в который будут добавляться варианты перевода слова
                translation_div = (response_soup.find_all('div', class_='middle_col')[2])
                translation_tr = (translation_div.find_all('tr')[2])
                translation_td = (translation_tr.find_all('td')[1])
                for a in translation_td.find_all('a', recursive=False, limit=4):  # набираем 4 варианта перевода
                    if re.findall(r"[А-Яа-я]", a.text):  # проверяем, есть ли перевод на русский язык, а не ссылка на
                        # другую форму слова на английском языке (напр., head up = heads-up), и если есть, то записываем
                        # этот перевод, а если нет, то идем дальше
                        translation_output.append(a.text)
                        translation_result = transcription_span_text + ' - ' + '; '.join(translation_output)
                        # записываем варианты перевода
                        translated_dict[word_to_translate] = translation_result
                    else:
                        continue
            except Exception:  # если совсем ничего не нашлось на обоих сайтах, пропускаем слово
                continue
    return translated_dict


# text_dict = {}  # создаем пустой словарь
# url = 'https://bigbangtrans.wordpress.com/'  # сохраняем в переменную ссылку на страницу
# url_response = rq.get(url)  # с помощью requests получаем ответ сервера страницы и клдаем в переменную r
# html = url_response.text  # достаем из ответа сервера исходный код страниц и кладем в переменную html
# soup = BeautifulSoup(html, 'html.parser')  # передаем исходный код BeautifulSoup
# for tag in soup.find_all('a'):  # итерируемся по всем тэгам a, чтобы найти тексты серий
#     if 'episode' not in tag['href']:  # проверяем, есть ли в ссылках внутри этих тэгов строка episode
#         continue
#     episode_name = tag.text  # сохраняем название серии
#     child_url = f'{tag["href"]}'  # формируем url страницы с текстов из двух частей
#     child_url_response = rq.get(child_url)  # сохраняем ответ сервера на запрос к странице с текстом
#     child_html = child_url_response.text  # сохраняем исходный код страницы из ответа сервера
#     child_soup = BeautifulSoup(child_html, 'html.parser')  # делаем из него суп
#     episode_text = []
#     for child_tag in child_soup.find_all('div', class_='entrytext'):  # находим и добавляем в список тексты серий
#         for content in child_tag.find_all('p'):
#             episode_text.append(content.text.strip().replace('\n\n', '\n'))
#     episode_text = '\n'.join(episode_text)
#     text_dict[episode_name] = episode_text  # добавляем в словарь текст серии в качестве ключа
#     with open("transcripts.txt", 'a', encoding='utf-8') as transcripts_file:  # записываем словарь в файл
#         transcripts_file.write(episode_name + '\n\n')
#         transcripts_file.write(episode_text + '\n\n\n')
#
# with open('transcripts.txt', 'r', encoding='utf-8') as transcript_file:
#     with open('preprocessed_transcripts.txt', 'a', encoding='utf-8') as preprocessed_transcript_file:
#         preprocessed_texts = '\n'.join(transcript_file.readlines())
#         result = preprocessed_texts.replace('\n\n', '\n')
#         result = result.replace('\n\n', '\n')
#         preprocessed_transcript_file.write(result)
#
#
# # Создаем словарь формата "(номер эпизода + номер серии, напр., 0101): текст серии, очищенный от лишних слов, фраз, \
# # которые не произносятся героями (напр., описание места действия или указание говорящего), знаков препинания, \
# # ненужных элементов и пробелов".
# with open('preprocessed_transcripts.txt', 'r', encoding='utf-8') as corpus:
#     document = corpus.read()
#     transcript_dict = {}
#     series_episode_pattern = 'Series (\\d{2}) Episode (\\d{2}) – [\\w\\W]*?\n([\\w\\W]*?)(?=\n^\n)'
#     series_episode_found_list = re.findall(series_episode_pattern, document, flags=re.MULTILINE)
#     for elem in series_episode_found_list:
#         key = elem[0] + elem[1]
#         new_document = elem[2]
#         new_document = re.sub(r'^[^:]*?\n', '', new_document, flags=re.MULTILINE)
#         new_document = re.sub(r'^(Scene:|Teleplay:|Story:|Written by) [\w\W]*?(?=$)', '', new_document,
#                               flags=re.MULTILINE)
#         new_document = re.sub(r'^[\w\W]*?: ', '', new_document, flags=re.MULTILINE)
#         new_document = re.sub(r'\([\w\W]*?\)[,: ]*', '', new_document, flags=re.MULTILINE)
#         new_document = re.sub(r'\s\s', ' ', new_document, flags=re.MULTILINE)
#         new_document = re.sub(r'\.\s\.\.', '…', new_document, flags=re.MULTILINE)
#         new_document = re.sub(r'(?<=[.!,?…])\s\.', '', new_document, flags=re.MULTILINE)
#         new_document = re.sub(r'\s\.', '.', new_document, flags=re.MULTILINE)
#         new_document = re.sub(r'%', ' percent', new_document, flags=re.MULTILINE)  # json "спотыкается" о символ %
#         transcript_dict[key] = new_document
#
#
# # Для облегчения дальнейшей работы со словарем, содержащим тексты серий, записываем его в файл
# with open('transcript_dictionary.txt', 'w', encoding='utf-8') as dict_to_fill:  # открываем нужный файл для записи
#     transcript_file = str(transcript_dict)  # превращаем словарь в строку
#     transcript_file = re.sub(r'\\xa0', '', transcript_file)  # удаляем из нее лишние случайные символы
#     transcript_file = re.sub(r'\s\s', ' ', re.sub(r'\\n', ' ', transcript_file))  # также заменяем лишние переносы
#     # строк на пробелы и двойные пробелы на одинарные
#     dict_to_fill.write(transcript_file)  # записываем словарь в файл
#
# # Извлекаем словарь с текстами серий из файла для последующего использования
# with open('transcript_dictionary.txt', 'r', encoding='utf-8') as file_to_use:  # открываем нужный файл для чтения
#     read_string = file_to_use.read()  # считываем файл
#     number_list = re.findall(r'(?<=\')[^:,][\w\W]*?(?=\')', read_string)  # находим все элементы вида 'текст'
#     v = iter(number_list)  # итерируем полученный список элементов
#     dict_to_use = {s: next(v) for s in v}  # создаем словарь, ключи которого - номера серий, а значения - их тексты
#
# with open('stopwords+words_already_known.txt', 'r', encoding='utf-8') as stop_file:  # открываем файл, в который
#     # добавлены стоп-слова и первые 1 500 слов из частотного списка для английского языка
#     read_file = stop_file.readlines()  # считываем файл
#     stopwords = re.findall(r'\'(\S*)\'', read_file[0])  # извлекаем из него список стоп-слов, которые не будут
#     # учитываться при поиске слов для перевода
#
# with open('translations_dict.txt', 'a', encoding='utf-8') as corp:  # открываем файл для записи посерийного словаря
#     keys = dict_to_use.keys()  # в качестве ключей берем ключи созданного выше словаря (т. е. номера серий)
#     for key in dict_to_use.keys():  # проходимся по каждому ключу
#         dict_to_use[key] = [word for word in get_words_to_be_translated(dict_to_use[key], stopwords)]  # получаем
# # слова, которые необходимо перевести для каждой серии, с помощью функции get_words_to_be_translated
#         dict_to_use[key] = get_translations(dict_to_use[key])  # получаем списки этих слов вместе с их переводами \
# # с помощью функции get_translations
#         regex = re.compile("\'[\\s?а-яА-я\']*?\'", re.S)
#         dict_to_use[key] = {k: regex.sub(lambda m: m.group().replace("\'", '\"', 2), v) for k, v in
#                             dict_to_use[key].copy().items()}  # заменяем одинарные кавычки на двойные в переводе
#         # слова, чтобы они не мешали дальнейшему считыванию элементов словаря
#     values = dict_to_use.values()  # в качестве значений берем полученные списки слов с их переводами
#     translation_dict = dict(zip(keys, values))  # создаем словарь, где ключи - это номера серий, а значения - списки \
#     # слов с переводами
#     corp.write(str(translation_dict))  # записываем словарь в файл для облегчения дальнейшей работы


# Извлекаем заранее составленный словарь для нужной серии из файла со словарями
with open('translations_dict.txt', 'r', encoding='utf-8') as string_to_read:  # открываем файл
    read_string = string_to_read.read()  # прочитываем его
    numbers_list = re.findall(r'(?<=\')\d{4}(?=\': {)', read_string)  # находим все номера серий
    translations_list = re.findall(r'(?<=\'\d{4}\': {)[\w\W]*?(?=})', read_string)  # находим все строки со словарями
    keys = [re.findall(r'(?<=\')[^:\s,][\w\W]*?(?=\':)', translation) for translation in translations_list]  # находим
    # в найденных строках все слова для каждой серии
    values = [re.findall(r'(?<=\': \')\s[\w\W]*?(?=\')', translation) for translation in translations_list]  # находим
    # в найденных строках все переводы этих слов
    tr_dicts = [dict(zip(keys[i], values[i])) for i in range(len(keys))]  # создаем словари для всех серий
    translations_dict = dict(zip(numbers_list, tr_dicts))  # создаем словарь вида {номер серии: словарь к серии}


# Просим пользователя указать сезон и номер серии для вывода словаря
series_episode = input("\nКакой сезон и какую серию хочешь посмотреть? Введи их без пробела, например, 0907: ")
print()
# Вывод словаря к серии
for key in translations_dict[series_episode].keys():
    print(key, translations_dict[series_episode][key], sep='')

# # Если нужно записать словарь к серии в файл:
# with open('episode_vocabulary.txt', 'a', encoding='utf-8') as vocabulary_file:
#     for key in translations_dict[series_episode].keys():  # выводим словарь для нужной серии
#         vocabulary_file.write(key)
#         vocabulary_file.write(' ')
#         vocabulary_file.write(translations_dict[series_episode][key])
#         vocabulary_file.write('\n')
#
# # Запись частотного словаря в файл для ускорения его вывода в дальнейшем
# with open('frequency_dictionary.txt', 'a', encoding='utf-8') as frequency_file:
#     words_to_count_list = []  # создаем список всех слов из словаря по сериалу, включая повторяющиеся
#     for value in translations_dict.values():
#         for keys in value:
#             words_to_count_list.append(keys)
#     words_to_count_set = list(set(words_to_count_list))  # превращаем этот список во множество для исключения \
#     # повторяющихся слов и сохраняем его в виде другого списка
#     frequency_dict = {word: words_to_count_list.count(word) for word in words_to_count_set}  # создаем частотный
#     # словарь, записывая каждое слово из списка без повторений и его частоту, подсчитанную в списке с повторениями
#     for word_counted in sorted(frequency_dict, key=frequency_dict.get, reverse=True):
#         for values in translations_dict.values():  # ищем перевод каждого слова в словарях к сериям,
#             if word_counted in values:  # ... если это слово в них встречается,
#                 word_translation = values.get(word_counted)  # а после того, как перевод нашелся,
#                 break  # прерываем поиск перевода, т. к. одно и то же слово может встречаться в разных сериях
#         frequency_file.write(str(frequency_dict[word_counted]))  # выводим частоту
#         # слова, само слово и его перевод для всех слов
#         if len(str(frequency_dict[word_counted])) == 3:
#             frequency_file.write('      ')
#         elif len(str(frequency_dict[word_counted])) == 2:
#             frequency_file.write('       ')
#         else:
#             frequency_file.write('        ')
#         frequency_file.write(word_counted)
#         frequency_file.write(word_translation)
#         frequency_file.write('\n')
#
#
# # Вывод частотного словаря по всему сериалу, состоящего из слов, которые нужно выучить
# print('\n\n')
# with open('frequency_dictionary.txt', 'r', encoding='utf-8') as word_frequency_file:
#     freq_file = word_frequency_file.readlines()
#     [print(line, end='') for line in freq_file]
#
# Добавление знакомых слов в список изученных для последующего удаления из переведенного набора слов
already_learned_words = input('\n\nДобавить слова в изученные (слова вводятся через пробел): ')
print('\n\n')
already_learned_words = already_learned_words.split()

# Удаление лишних (добавленных в изученные) слов из переведенного набора слов
already_learned_words_set = frozenset(already_learned_words)
translation_list = [key for key in translations_dict[series_episode].keys() if key not in already_learned_words_set]
for key in translation_list:
    print(key, translations_dict[series_episode][key], sep='')
