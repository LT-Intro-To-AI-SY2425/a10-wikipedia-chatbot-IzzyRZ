import re, string, calendar
from wikipedia import WikipediaPage
import wikipedia
from bs4 import BeautifulSoup
from nltk import word_tokenize, pos_tag, ne_chunk
from nltk.tree import Tree
from match import match
from typing import List, Callable, Tuple, Any, Match


def get_page_html(title: str) -> str:
    """Gets html of a wikipedia page

    Args:
        title - title of the page

    Returns:
        html of the page
    """
    results = wikipedia.search(title)
    return WikipediaPage(results[0]).html()


def get_first_infobox_text(html: str) -> str:
    """Gets first infobox html from a Wikipedia page (summary box)

    Args:
        html - the full html of the page

    Returns:
        html of just the first infobox
    """
    soup = BeautifulSoup(html, "html.parser")
    results = soup.find_all(class_="infobox")

    if not results:
        raise LookupError("Page has no infobox")
    return results[0].text


def clean_text(text: str) -> str:
    """Cleans given text removing non-ASCII characters and duplicate spaces & newlines

    Args:
        text - text to clean

    Returns:
        cleaned text
    """
    only_ascii = "".join([char if char in string.printable else " " for char in text])
    no_dup_spaces = re.sub(" +", " ", only_ascii)
    no_dup_newlines = re.sub("\n+", "\n", no_dup_spaces)
    return no_dup_newlines


def get_match(
    text: str,
    pattern: str,
    error_text: str = "Page doesn't appear to have the property you're expecting",
) -> Match:
    """Finds regex matches for a pattern

    Args:
        text - text to search within
        pattern - pattern to attempt to find within text
        error_text - text to display if pattern fails to match

    Returns:
        text that matches
    """
    p = re.compile(pattern, re.DOTALL | re.IGNORECASE)
    match = p.search(text)

    if not match:
        raise AttributeError(error_text)
    return match


def get_polar_radius(planet_name: str) -> str:
    """Gets the radius of the given planet

    Args:
        planet_name - name of the planet to get radius of

    Returns:
        radius of the given planet
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(planet_name)))
    pattern = r"(?:Polar radius.*?)(?: ?[\d]+ )?(?P<radius>[\d,.]+)(?:.*?)km"
    error_text = "Page infobox has no polar radius information"
    match = get_match(infobox_text, pattern, error_text)
    radiusInfo = f"the polar radius of {planet_name} is {match.group('radius')} km"
    return radiusInfo


def get_birth_date(name: str) -> str:
    """Gets birth date of the given person

    Args:
        name - name of the person

    Returns:
        birth date of the given person
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(name)))
    pattern = r"(?:Born\D*)(?P<birth>\d{4}-\d{2}-\d{2})"
    error_text = (
        "Page infobox has no birth information (at least none in xxxx-xx-xx format)"
    )
    match = get_match(infobox_text, pattern, error_text)
    birthInfo = f"{name} was born on this date: {match.group('birth')}"
    return birthInfo

def get_population_size(place: str) -> str:
    """Gets the population size of the given place
    
    Args: 
        place - name of the place
        
    Returns:
        population size of the given place
    """
    infobox_text = clean_text(get_first_infobox_text(get_page_html(place)))
    print(infobox_text)
    pattern = r"Population\D*\d{4}[)\[\d]*\D*(?P<Population>[\d,]+)" 
    error_text = (
        "Page infobox has no population information"
    )
    match = get_match(infobox_text, pattern, error_text)
    popInfo = f"{place} has a population of {match.group('Population')}"
    return popInfo
    
def get_establish_year(thing: str) -> str:
    infobox_text = clean_text(get_first_infobox_text(get_page_html(thing)))
    print(infobox_text)
    pattern = r"Established[\n\s]*(?P<time>[\d\w\s,]+)[;\(]+"
    error_text = (
        "Page infobox has no establishment information (at least not in the 'established' format)"
    )
    match = get_match(infobox_text, pattern, error_text)
    estInfo = f"{thing} was established in {match.group('time')}"
    return estInfo

def get_ugrad_pop(school: str) -> str:
    infobox_text = clean_text(get_first_infobox_text(get_page_html(school)))
    print(infobox_text)
    pattern = r"Undergraduates[\n\s]*(?P<Ugrad>[\d,]+)"
    error_text = (
        "Page infobox has no information on undergraduate population"
    )
    match = get_match(infobox_text, pattern, error_text)
    ugradInfo = f"{school} has an undergraduate population of {match.group('Ugrad')}"
    return ugradInfo
    
# below are a set of actions. Each takes a list argument and returns a list of answers
# according to the action and the argument. It is important that each function returns a
# list of the answer(s) and not just the answer itself.


def birth_date(matches: List[str]) -> List[str]:
    """Returns birth date of named person in matches

    Args:
        matches - match from pattern of person's name to find birth date of

    Returns:
        birth date of named person
    """
    return [get_birth_date(" ".join(matches))]


def polar_radius(matches: List[str]) -> List[str]:
    """Returns polar radius of planet in matches

    Args:
        matches - match from pattern of planet to find polar radius of

    Returns:
        polar radius of planet
    """
    return [get_polar_radius(matches[0])]

def population_size(matches: List[str]) -> List[str]:
    return [get_population_size(matches[0])]

def establish_year(matches: List[str]) -> List[str]:
    return [get_establish_year(matches[0])]

def ugrad_pop(matches: List[str]) -> List[str]:
    return [get_ugrad_pop(matches[0])]

# dummy argument is ignored and doesn't matter
def bye_action(dummy: List[str]) -> None:
    raise KeyboardInterrupt


# type aliases to make pa_list type more readable, could also have written:
# pa_list: List[Tuple[List[str], Callable[[List[str]], List[Any]]]] = [...]
Pattern = List[str]
Action = Callable[[List[str]], List[Any]]

# The pattern-action list for the natural language query system. It must be declared
# here, after all of the function definitions
pa_list: List[Tuple[Pattern, Action]] = [
    ("when was % born".split(), birth_date),
    ("what is the polar radius of %".split(), polar_radius),
    ("When was % established".split(), population_size),
    ("what is the population of %".split(), population_size),
    ("what year was % established".split(), establish_year),
    ("what is the undergraduate population of %".split(), ugrad_pop),
    (["bye"], bye_action),
]


def search_pa_list(src: List[str]) -> List[str]:
    """Takes source, finds matching pattern and calls corresponding action. If it finds
    a match but has no answers it returns ["No answers"]. If it finds no match it
    returns ["I don't understand"].

    Args:
        source - a phrase represented as a list of words (strings)

    Returns:
        a list of answers. Will be ["I don't understand"] if it finds no matches and
        ["No answers"] if it finds a match but no answers
    """
    for pat, act in pa_list:
        mat = match(pat, src)
        if mat is not None:
            answer = act(mat)
            return answer if answer else ["No answers"]

    return ["I don't understand"]


def query_loop() -> None:
    """The simple query loop. The try/except structure is to catch Ctrl-C or Ctrl-D
    characters and exit gracefully"""
    print("Welcome to the wikipedia database!\n")
    while True:
        try:
            print()
            query = input("Your query? ").replace("?", "").lower().split()
            answers = search_pa_list(query)
            for ans in answers:
                print(ans)

        except (KeyboardInterrupt, EOFError):
            break
        except(AttributeError):
            print("Page does not match pattern!")

    print("\nSo long!\n")


# uncomment the next line once you've implemented everything are ready to try it out
query_loop()
#Transcript:
#Welcome to the wikipedia database!


# Your query? What is the population of Quebec?
# Quebec
# Qu bec (French)Province
# FlagCoat of armsMotto(s): Je me souviens (French)"I remember"
# BC
# AB
# SK
# MB
# ON
# QC
# NB
# PE
# NS
# NL
# YT
# NT
# NU
# Coordinates: 52 N 72 W / 52 N 72 W / 52; -72[1]CountryCanadaBefore confederationCanada EastConfederationJuly 1, 1867 (1st, with New Brunswick, Nova Scotia, Ontario)CapitalQuebec CityLargest cityMontreal Largest metroGreater Montreal
# Government TypeParliamentary constitutional monarchy Lieutenant GovernorManon Jeannotte PremierFran ois Legault LegislatureNational Assembly of QuebecFederal representationParliament of CanadaHouse seats78 of 343 (22.7%)Senate seats24 of 105 (22.9%)
# Area Total1,542,056 km2 (595,391 sq mi) Land1,365,128 km2 (527,079 sq mi) Water176,928 km2 (68,312 sq mi) 11.5% Rank2nd 15.4% of CanadaPopulation (2021) 
# Total8,501,833[2] Estimate (Q1 2025)9,111,629[3] Rank2nd Density6.23/km2 (16.1/sq mi)Demonym(s)in English: Quebecer, Quebecker, Qu b cois in French: Qu b cois (m),[4] Qu b coise (f)[4] Official languagesFrench[5]
# GDP Rank2nd Total (2022)C$552.737 billion[6] Per capitaC$63,651 (9th)HDI HDI (2019)0.916[7] Very high (9th)Time zoneUTC 05:00 (Eastern Time Zone for most of the province[8]) Summer (DST)UTC 04:00Canadian postal abbr.QC[9]Rankings include all provinces and territories
# quebec has a population of 8,501,833

# Your query? What year was Harvard University established
# Harvard UniversityCoat of armsLatin: Universitas Harvardiana[1][2]Former namesHarvard CollegeMottoVeritas (Latin)[3]Motto in English"Truth"TypePrivate research universityEstablishedOctober 28, 1636 (388 years ago) (1636-10-28)[4]FounderMassachusetts General CourtAccreditationNECHEAcademic affiliationsAAUCOFHENAICUUArcticURASpace-grantEndowment$50.7 billion (2023)[5][6]PresidentAlan GarberProvostJohn F. Manning[7]Academic staff~2,400 faculty members (and >10,400 academic appointments in affiliated teaching hospitals)[8]Students21,278 (fall 2023)[9]Undergraduates7,110 (fall 2023)[9]Postgraduates14,168 (fall 2023)[9]LocationCambridge, Massachusetts, United States42 22 28 N 71 07 01 W / 42.37444 N 71.11694 W / 42.37444; -71.11694CampusMidsize city[10], 209 acres (85 ha)NewspaperThe Harvard CrimsonColorsCrimson, white, and black[11] NicknameCrimsonSporting affiliationsNCAA Division I FCS Ivy LeagueECAC HockeyNEISACWPAIRAEAWRCEARCEISAMascotJohn HarvardWebsitewww.harvard.edu
# harvard university was established in October 28, 1636 

# Your query? What is the undergraduate population of university of illinois at urbana-champaign?
# University of IllinoisUrbana-ChampaignFormer namesIllinois Industrial University (1867 1885)University of Illinois (1885 1982)University of Illinois at Urbana-Champaign (1982 2021)[1]Motto"Learning & Labor"TypePublic land-grant research universityEstablished1867; 158 years ago (1867)Parent institutionUniversity of Illinois SystemAccreditationHLCAcademic affiliationsAAUURASea-grantSpace-grantEndowment$3.38 billion (2023)(system-wide)[2]Budget$7.7 billion (2023) (system-wide)[3]ChancellorRobert J. Jones[4]PresidentTimothy L. Killeen[5]ProvostJohn Coleman[6]Academic staff2,548Administrative staff8,803[7]Students59,238 (2024)[8]Undergraduates37,140 (2024)[8]Postgraduates20,765 (2024)[8]LocationUrbana-Champaign, Illinois, United StatesCampusSmall city[10], 6,370 acres (2,578 ha)[9]NewspaperThe Daily IlliniColorsOrange and blue[11] NicknameFighting IlliniSporting affiliationsNCAA Division I FBS Big TenWebsiteillinois.edu
# university of illinois at urbana-champaign has an undergraduate population of 37,140

# Your query? What is the population of paris?
# ParisCapital city, commune and departmentEiffel Tower and the Seine from Tour Saint-JacquesNotre-DameSacr -C urPanth onArc de TriomphePalais GarnierThe Louvre
# FlagCoat of armsMotto(s): Fluctuat nec mergitur"Tossed by the waves but never sunk"Location of Paris
# ParisShow map of FranceParisShow map of le-de-France (region)Coordinates: 48 51 24 N 2 21 8 E / 48.85667 N 2.35222 E / 48.85667; 2.35222CountryFranceRegion le-de-FranceDepartmentParisArrondissementNoneIntercommunalityM tropole du Grand ParisSubdivisions20 arrondissementsGovernment Mayor (2020 2026) Anne Hidalgo[1] (PS)Area1[2]105.4 km2 (40.7 sq mi) Urban (2021[2])2,824.2 km2 (1,090.4 sq mi) Metro (2021[2])18,940.7 km2 (7,313.0 sq mi)Population (Jan 2025[3])2,048,472 Rank9th in Europe1st in France Density19,000/km2 (50,000/sq mi) Urban (2021)10,890,751[2] Metro (2021)13,171,056[2]Demonym(s)Parisian(s) (en) Parisien(s) (masc.), Parisienne(s) (fem.) (fr), Parigot(s) (masc.), "Parigote(s)" (fem.) (fr, colloquial)Time zoneUTC+01:00 (CET) Summer (DST)UTC+02:00 
# (CEST)INSEE/Postal code75056 /75001 75020, 75116Elevation28 131 m (92 430 ft) (avg. 78 m or 256 ft)Websiteparis.fr1 French Land Register data, which excludes lakes, ponds, glaciers > 1 km2 (0.386 sq mi or 247 acres) and river estuaries.
# paris has a population of 048,472 Originally got the wrong number for paris because of an oversight with the regular expression (it omitted the first number because it counted it as part of the citation because there were no spaces in between the citation and the number) so we fixed it by making sure the citation set does not include a closing bracket so this mistake cannot happen anymore.
# Fixed version:
# Your query? what is the population of paris
# ParisCapital city, commune and departmentEiffel Tower and the Seine from Tour Saint-JacquesNotre-DameSacr -C urPanth onArc de TriomphePalais GarnierThe Louvre     
# FlagCoat of armsMotto(s): Fluctuat nec mergitur"Tossed by the waves but never sunk"Location of Paris
# ParisShow map of FranceParisShow map of le-de-France (region)Coordinates: 48 51 24 N 2 21 8 E / 48.85667 N 2.35222 E / 48.85667; 2.35222CountryFranceRegion le-de-FranceDepartmentParisArrondissementNoneIntercommunalityM tropole du Grand ParisSubdivisions20 arrondissementsGovernment Mayor (2020 2026) Anne Hidalgo[1] (PS)Area1[2]105.4 km2 (40.7 sq mi) Urban (2021[2])2,824.2 km2 (1,090.4 sq mi) Metro (2021[2])18,940.7 km2 (7,313.0 sq mi)Population (Jan 2025[3])2,048,472 Rank9th in Europe1st in France Density19,000/km2 (50,000/sq mi) Urban (2021)10,890,751[2] Metro (2021)13,171,056[2]Demonym(s)Parisian(s) (en) Parisien(s) (masc.), Parisienne(s) (fem.) (fr), Parigot(s) (masc.), "Parigote(s)" (fem.) (fr, colloquial)Time zoneUTC+01:00 (CET) Summer (DST)UTC+02:00 (CEST)INSEE/Postal code75056 /75001 75020, 75116Elevation28 131 m (92 430 ft) (avg. 78 m or 256 ft)Websiteparis.fr1 French Land Register data, which excludes lakes, ponds, glaciers > 1 km2 (0.386 sq mi or 247 acres) and river estuaries.
# paris has a population of 2,048,472

# Your query? What is the population of miami university of ohio
# Miami UniversityMottoProdesse Quam Conspici (Latin)Motto in English"To accomplish without being conspicuous"[1]TypePublic research universityEstablishedFebruary 2, 1809; 216 years ago (1809-02-02)Parent institutionUniversity System of OhioAccreditationHLCAcademic affiliationsGC3Space-grantEndowment$814 million (2024) [2]PresidentGregory Crawford[3]ProvostElizabeth Mullenix[4]Academic staff1,106 (fall 2023)[5]Students18,618 (fall 2023)[5]Undergraduates16,478 (fall 2023)[5]Postgraduates2,140 (fall 2023)[5]LocationOxford, Ohio, United States39 30 43 N 84 44 05 W / 39.511905 N 84.734674 W / 39.511905; -84.734674CampusFringe town[6], 2,138 acres (8.65 km2)Other campusesHamiltonMiddletownWest ChesterDifferdangeNewspaperThe Miami StudentColorsRed and white[7][8] NicknameRedHawksSporting affiliationsNCAA Division I FBS MACNCHCMascotSwoop the RedHawkWebsitemiamioh.edu
# Page does not match pattern!

# Your query? what year was miami university of ohio established?
# Miami UniversityMottoProdesse Quam Conspici (Latin)Motto in English"To accomplish without being conspicuous"[1]TypePublic research universityEstablishedFebruary 2, 1809; 216 years ago (1809-02-02)Parent institutionUniversity System of OhioAccreditationHLCAcademic affiliationsGC3Space-grantEndowment$814 million (2024) [2]PresidentGregory Crawford[3]ProvostElizabeth Mullenix[4]Academic staff1,106 (fall 2023)[5]Students18,618 (fall 2023)[5]Undergraduates16,478 (fall 2023)[5]Postgraduates2,140 (fall 2023)[5]LocationOxford, Ohio, United States39 30 43 N 84 44 05 W / 39.511905 N 84.734674 W / 39.511905; -84.734674CampusFringe town[6], 2,138 acres (8.65 km2)Other campusesHamiltonMiddletownWest ChesterDifferdangeNewspaperThe Miami StudentColorsRed and white[7][8] NicknameRedHawksSporting affiliationsNCAA Division I FBS MACNCHCMascotSwoop the RedHawkWebsitemiamioh.edu
# miami university of ohio was established in February 2, 1809

# Your query? What is the undergraduate population of indiana university? 
# Indiana UniversityLatin: Indianensis UniversitasMottoLux et Veritas(Light and Truth)TypePublic university systemEstablishedJanuary 20, 1820; 205 years ago (1820-01-20)Endowment$3.56 billion (2023)[1] (system-wide)PresidentPamela WhittenAcademic staff8,733 university-wide[2]Students110,436 university-wide[2]Undergraduates89,176 university-wide[2]Postgraduates21,260 university-wide[2]LocationBloomington, IndianaIndianapolis, Indiana39 10 N 86 30 W / 39.167 
# N 86.500 W / 39.167; -86.500Campus3,640 acres (14.7 km2) across 9 campuses[2]ColorsCream and Crimson Websitewww.indiana.edu
# indiana university has an undergraduate population of 89,176