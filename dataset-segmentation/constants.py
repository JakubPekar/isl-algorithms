# A320
from typing import List


""""""""""""""""""""""""
""""""""" A320 """""""""
""""""""""""""""""""""""

SIGNALS_PATH = 'real-data/A320'
OUTPUT_PATH = 'real-data/A320'
ROOM_DIMENSIONS = [8.2, 6.85, 3]
MIC_CENTROID = [0.7, 3.43, 1.5]

POSITION_MAP = {
    1: [2.88, 0.33, 1.62],
    2: [2.81, 1.75, 1.62],
    3: [2.86, 3.36, 1.62],
    4: [2.78, 5.01, 1.62],
    5: [2.74, 6.35, 1.62],
    6: [3.94, 6.34, 1.62],
    7: [3.99, 4.97, 1.62],
    8: [3.99, 3.53, 1.62],
    9: [3.99, 1.66, 1.62],
    10: [4.06, 0.37, 1.62],
    11: [5.29, 0.35, 1.62],
    12: [5.19, 1.66, 1.62],
    13: [5.3, 3.38, 1.62],
    14: [5.25, 4.91, 1.62],
    15: [5.22, 6.32, 1.62],
    16: [6.41, 6.33, 1.62],
    17: [6.49, 4.87, 1.62],
    18: [6.52, 3.43, 1.62],
    19: [6.45, 1.48, 1.62],
    20: [6.53, 0.26, 1.62],
    21: [7.81, 0.37, 1.62],
    22: [7.84, 1.68, 1.62],
    23: [7.87, 3.32, 1.62],
    24: [7.84, 4.85, 1.62],
    25: [7.89, 6.54, 1.62],
}




""""""""""""""""""""""""
"""""""""  KD  """""""""
""""""""""""""""""""""""

SIGNALS_PATH = 'real-data/KD'
OUTPUT_PATH = 'real-data/KD'
ROOM_DIMENSIONS = [13.95, 11.8, 7]
MIC_CENTROID = [1, 5.9, 1.5]


def get_position(key: int, index: int) -> List[float]:
    x = (key - 1) // 9 + 1
    y = (key - 1) % 9 - 4
    if not x % 2:
        y *= -1
    z = -0.35 if 2 * key < index - 10 else 0.1
    return [x, y, z]


LABELS_MAP = [
    "Včera jsem si uvařil vynikající guláš.",
    "Na dnešním programu mám schůzku s klientem.",
    "Tento rok si plánujeme udělat dovolenou v Itálii.",
    "Rád poslouchám klasickou hudbu.",
    "Zítra ráno musím jít na kontrolu k zubaři.",
    "Můj nejoblíbenější sport je plavání.",
    "Když jsem byl malý, chtěl jsem být astronautem.",
    "V naší firmě jsou v současné době volná pracovní místa.",
    "Na výletě jsem si koupil novou kabelku.",
    "Před dvěma týdny jsem si koupil nový počítač.",
    "V neděli ráno si vždycky připravuji vajíčka na snídani.",
    "Do práce jezdím každý den autobusem.",
    "Tento film jsem viděl už několikrát.",
    "Večer si rád čtu knihy.",
    "Vím, že jsem udělal chybu a omlouvám se.",
    "Dneska mě celý den bolí hlava.",
    "Našel jsem na ulici peněženku a odevzdal ji na policii.",
    "Rád cestuji a poznávám nové kultury.",
    "Moje oblíbená barva je modrá.",
    "Dneska jsem šel na procházku do lesa.",
    "Byl jsem na návštěvě u své babičky.",
    "V pátek jdeme s kamarády na pivo.",
    "Rád si poslechnu dobrý podcast.",
    "Musím si objednat nové brýle.",
    "Večer si chystám palačinky se sirupem.",
    "Rád bych si objednal jednu kávu s mlékem.",
    "Večer jsem se setkal se svými přáteli v restauraci.",
    "Dneska bylo krásné počasí, tak jsme se rozhodli jít na procházku.",
    "Včera jsem si koupil novou knihu od mého oblíbeného autora.",
    "Když jsem byl malý, rád jsem hrával fotbal s kamarády.",
    "Příští týden jedu na dovolenou do Itálie.",
    "Potřebuji koupit nové boty, mé staré už jsou opravdu ošoupané.",
    "Na výletě jsme navštívili krásný zámek s velkou zahradou.",
    "Včera večer jsem se učil na zkoušku z matematiky.",
    "Tento film se mi opravdu líbil, měl dobrý příběh a herecké obsazení.",
    "Rád poslouchám různé druhy hudby, ale nejvíce preferuji rockovou.",
    "Dneska ráno jsem si udělal omeletu s houbami a paprikou.",
    "Mám rád jarní období, kdy se všechno začíná zelenat a kvést.",
    "Dneska jsem měl velmi náročný den v práci.",
    "Před pár dny jsem navštívil koncert mé oblíbené kapely.",
    "Miluji cestování a objevování nových kultur a zemí.",
    "Vždycky si dávám k snídani ovesné vločky s ovocem a jogurtem.",
    "Minulou sobotu jsem strávil celý den u vody a opaloval se.",
    "Každý večer před spaním si rád přečtu několik stránek knihy.",
    "V Praze je mnoho krásných památek, které stojí za návštěvu.",
    "Přišla mi pozvánka na svatbu mé sestry, těším se na ni velmi.",
    "Rád sportuji, často běhám nebo jezdím na kole.",
    "Dneska jsem byl s rodinou na obědě v italské restauraci.",
    "Byl jsem na koupališti s kamarády, bylo to velmi zábavné.",
    "Rád poslouchám podcasty o různých tématech, jako je například technologie nebo psychologie.",
    "Dnes je krásný slunečný den.",
    "Miluji českou kuchyni.",
    "Praha je hlavní město České republiky.",
    "Vím, jak mluvit česky.",
    "Rád bych ochutnal tradiční české pivo.",
    "Mám rád české filmy.",
    "Česká literatura je velmi bohatá a zajímavá.",
    "Včera jsem navštívil staré město v Olomouci.",
    "Jezero Lipno je populární turistickou destinací.",
    "Naše země je bohatá na kulturní památky.",
    "Můj nejoblíbenější sport je hokej.",
    "Tento hotel má úžasný výhled na hory.",
    "S českou hospodou si nijak nelámou hlavu.",
    "Tento rok je počasí opravdu nevyzpytatelné.",
    "Moje oblíbené jídlo je tradiční svíčková na smetaně.",
    "Rád bych se naučil hrát na akordeon.",
    "V zimě rád jezdím na lyže do Krkonoš.",
    "Česká republika má mnoho krásných hradů a zámků.",
    "Tento film vyhrál cenu za nejlepší scénář.",
    "Můj oblíbený český spisovatel je Karel Čapek.",
    "Naši hokejisté vyhráli zlatou medaili na olympijských hrách.",
    "Pražský orloj je jednou z nejznámějších turistických atrakcí.",
    "Dnes večer jdeme na koncert do Lucerny.",
    "Český jazyk má mnoho složitých pravidel.",
    "V poslední době je v Česku stále populárnější cykloturistika.",
    "Dnes je krásné počasí.",
    "Mám rád českou kuchyni.",
    "Byl jsem v Praze minulý víkend.",
    "Tento film se mi nelíbil.",
    "Rád poslouchám českou hudbu.",
    "Můj nejoblíbenější sport je hokej.",
    "Tento knižní obchod má skvělý výběr.",
    "Už jsem několikrát navštívil Karlovy Vary.",
    "Potřebuju koupit nový počítač.",
    "V zimě lyžuji v Krkonoších.",
    "Moje oblíbené jídlo je guláš.",
    "Dnes jsem měl těžký den v práci.",
    "Rád trávím volný čas s rodinou.",
    "Můj nejoblíbenější český herec je Vlastimil Brodský.",
    "Cestování je moje vášeň.",
    "Dnes jsem se cítil unavený.",
    "Mám rád procházky v přírodě.",
    "Praha je nádherné město.",
    "Tento čaj je skvělý.",
    "Můj nejoblíbenější český spisovatel je Karel Čapek.",
    "Učím se číst a psát.",
    "V poslední době jsem měl hodně práce.",
    "Rád si čtu knihy o historii.",
    "Dnes jsem si uvařil vynikající večeři.",
    "Tento hotel má fantastický výhled na město.",
    "Dneska je hezký den.",
    "Rád bych si objednal kávu.",
    "Mám rád českou kuchyni.",
    "Kde je nejbližší bankomat?",
    "Učím se česky.",
    "Vítejte v Praze!",
    "Byla jsi někdy na Karlově mostě?",
    "Rád bych navštívil hrad Karlštejn."
]





""""""""""""""""""""""""
""""""""" C511 """""""""
""""""""""""""""""""""""

SIGNALS_PATH = 'real-data/C511'
OUTPUT_PATH = 'real-data/C511'
ROOM_DIMENSIONS = [8.15, 5.3, 3.25]
MIC_CENTROID = [0.37, 2.7, 1.5]

POSITION_MAP = {
        1: [1.95, 4.86],
        2: [1.80, 3.03],
        3: [3.14, 4.29],
        4: [3.18, 3.03],
        5: [3.18, 1.64],
        6: [4.26, 5.02],
        7: [4.33, 3.61],
        8: [4.33, 2.35],
        9: [5.54, 4.40],
        10: [5.33, 3.03],
        11: [5.54, 1.66],
        12: [7.04, 3.53],
        13: [7.02, 2.20],
    }

def get_position(key: int, index: int) -> List[float]:
    x, y = POSITION_MAP[key]
    x, y = x - 0.37, y - 2.7
    z = -0.37 if 2 * key < index - 5 else 0.12
    return [x, y, z]

LABELS_MAP = []







""""""""""""""""""""""""
""""""""" C525 """""""""
""""""""""""""""""""""""

SIGNALS_PATH = 'real-data/C525'
OUTPUT_PATH = 'ISL-Dataset/C525'
ROOM_DIMENSIONS = [9, 4.90, 3.25]
MIC_CENTROID = [0.38, 2.37, 1.5]

POSITION_MAP = {
        1: [3.32, 0.29],
        2: [3.21, 1.57],
        3: [3.18, 2.85],
        4: [4.46, 0.90],
        5: [4.39, 2.14],
        6: [4.46, 3.42],
        7: [5.60, 0.50],
        8: [5.67, 1.72],
        9: [5.71, 3.03],
        10: [6.98, 1.03],
        11: [7.01, 2.37],
        12: [6.99, 3.63],
        13: [8.49, 1.77],
        14: [8.44, 3.01],
    }

def get_position(key: int, index: int):
    x, y = POSITION_MAP[key]
    x, y = x - 0.37, y - 2.7
    z = -0.37 if 2 * key < index - 5 else 0.12
    return np.round([x, y, z], decimals=2).tolist()

LABELS_MAP = []