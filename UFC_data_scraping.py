# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.2
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# + [markdown] colab_type="text" id="view-in-github"
# <a href="https://colab.research.google.com/github/tylerlum/ufc_automated_scoring_system/blob/main/UFC_Automated_Scoring_System.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# + [markdown] id="TPiJGcdUYaoU"
# # UFC Data Scraping
#
# The goal of this notebook is to:
# * Explore the FightMetrics webpage to scrape the fight and fighter information we need
# * Preprocess the data
# * Store the fight and fighter data into csv files
#
# Functional as of April 2021

# + [markdown] id="Qx-IhTkrwliQ"
# ## Set parameters for dataset creation
#
# NUM_EVENTS_INPUT: Integer number of UFC events to get fights from or "All" for all events. There are about 10 fights per event.
#
# DATA_MODE_INPUT: Either "Summary" or "Round by Round". Either get data with columns that are summaries of the whole fight, or summaries round-by-round (more columns).

# + cellView="form" id="tYkaAlJNfhul"
# NUM_EVENTS_INPUT = "All"  #@param {type:"string"}
NUM_EVENTS_INPUT = "20"  #@param {type:"string"}
DATA_MODE_INPUT = "Summary"  #@param {type:"string"}
# -

NUM_EVENTS = None if NUM_EVENTS_INPUT == "All" else int(NUM_EVENTS_INPUT)
ROUND_BY_ROUND = (DATA_MODE_INPUT == "Round by Round")

# + [markdown] id="7897ryXiaoCv"
# ## Get information about all fighters

# + id="ioQESt2oZPXz"
import pandas as pd
from tqdm import tqdm
import numpy as np
import re
from string import ascii_lowercase


# + id="0WtkXEs0LNry"
def get_all_fighters():
    '''Get pandas table of all UFC fighters (Name, Height, Weight, Reach, Record, etc.)'''
    all_fighters_tables = []
    for c in tqdm(ascii_lowercase):
        all_fighters_url = f"http://ufcstats.com/statistics/fighters?char={c}&page=all"
        all_fighters_table = pd.read_html(all_fighters_url)[0]
        all_fighters_tables.append(all_fighters_table)

    all_fighters = pd.concat(all_fighters_tables)
    return all_fighters


# + colab={"base_uri": "https://localhost:8080/", "height": 216} id="AkCCSiuUa4lu" outputId="de1fcb58-1d27-4f12-8dd7-5644ddfb2ec6"
ALL_FIGHTERS = get_all_fighters()
ALL_FIGHTERS.head()

# + colab={"base_uri": "https://localhost:8080/"} id="Zoayc5Ad3tKm" outputId="629230e2-28a8-4ff8-d72f-1fa366f4f2fb"
ALL_FIGHTERS.dtypes

# + [markdown] id="bIDzcIHz3mpC"
# ## Clean fighter data
#
# TODO: Convert height, weight, reach to floats.

# + colab={"base_uri": "https://localhost:8080/", "height": 198} id="gdNWtyaB5cTi" outputId="db88f028-18b9-460d-e998-3a4c887c085f"
ALL_FIGHTERS = ALL_FIGHTERS.replace("^-+", np.nan, regex=True)  # Replace -- and --- with nan
ALL_FIGHTERS.dropna(subset=["First", "Last"], how='all')  # Remove rows with no name
ALL_FIGHTERS.head()


# + [markdown] id="HBvmzviJ625s"
# ## Helper functions

# + id="_KC8TRhZSW58"
def get_fighters(fighters_string):
    '''Parses string containing two fighter names. Uses ALL_FIGHTERS global to remove ambiguity in parsing. Returns each fighter name
       Eg. "Robert Whittaker Kelvin Gastelum" => ("Robert Whittaker", "Kelvin Gastelum")'''
    for i, row in ALL_FIGHTERS.iterrows():
        fighter_name = f'{row["First"]} {row["Last"]}'
        if fighters_string.startswith(fighter_name):
            first_fighter = fighter_name
            second_fighter = fighters_string[len(fighter_name)+1:]
            break
    return first_fighter, second_fighter

def remove_duplicates_keep_order(list_):
    '''Removes duplicates while keeping same order'''
    return list(dict.fromkeys(list_))


# + [markdown] id="hYGOutwI-g9H"
# ## Get a list of all UFC events

# + id="UqZSSQ8r-rMm"
from urllib.request import urlopen
from string import ascii_uppercase
from dateutil import parser
from datetime import datetime

# + id="t-X7tWV_NiBo"
ALL_PAST_EVENTS_URL = "http://ufcstats.com/statistics/events/completed?page=all"


# + id="JOvgx_9Z1SOv"
def get_all_events(all_past_events_url):
    '''Takes in URL to all past events. Returns list of urls, each one representing a UFC event'''
    all_past_events_html = urlopen(all_past_events_url).read().decode("utf-8")
    
    # Regex for "http://ufcstats.com/events-details/<alphanumeric>"
    # Eg. "http://ufcstats.com/event-details/27541033b97c076d"
    pattern = "\"http://ufcstats.com/event-details/[a-zA-Z0-9_]+\""
    all_urls = re.findall(pattern, all_past_events_html)

    # Remove quotes and duplicates
    all_urls = [url.strip("\"") for url in all_urls]
    all_urls = remove_duplicates_keep_order(all_urls)
    return all_urls


# + colab={"base_uri": "https://localhost:8080/"} id="zzxASuh8_Vnh" outputId="74c88906-ed19-4754-9b28-dde55d6bba45"
# Events
ALL_EVENT_URLS = get_all_events(ALL_PAST_EVENTS_URL)
print(f"Got {len(ALL_EVENT_URLS)} events")
print()

print("Removing the most recent event, since it might not have happened yet")
ALL_EVENT_URLS = ALL_EVENT_URLS[1:]
print(f"Now got {len(ALL_EVENT_URLS)} events")
print(ALL_EVENT_URLS)


# + [markdown] id="QoU8LAPK_dbQ"
# ## Get a list of UFC fights
#
# TODO: Right now only sees if result is win. Else sets winner to None. See if this can be improved.

# + id="2nDi4mo6CLn_"
def get_all_fights_in_event(past_event_url, get_results=False):
    '''Takes in a single URL to a past event.
       If get_results=True, returns fight_urls, winners, methods
       else, return fight_urls'''
    # Regex for "http://ufcstats.com/events-details/<alphanumeric>"
    # Eg. "http://ufcstats.com/fight-details/f67aa0b16e16a9ea"
    past_event_html = urlopen(past_event_url).read().decode("utf-8")
    pattern = "\"http://ufcstats.com/fight-details/[a-zA-Z0-9_]+\""
    fight_urls = re.findall(pattern, past_event_html)

    # Remove quotes and duplicates
    fight_urls = [url.strip("\"") for url in fight_urls]
    fight_urls = remove_duplicates_keep_order(fight_urls)

    # Get the winner and method (dec or KO or sub) of each fight
    past_event_table = pd.read_html(past_event_url)[0]  # Will be length 1 list
    winners, methods = [], []
    for _, row in past_event_table.iterrows():
        # TODO: Improve this processing of result
        result = row["W/L"].split(' ')[0]
        if result == "win":
            winner, _ = get_fighters(row["Fighter"])
        else:
            winner = None
        winners.append(winner)
        methods.append(row["Method"])

    if get_results:
        return fight_urls, winners, methods
    else:
        return fight_urls


# + id="7_pNUnyMPxkM"
def get_all_fights(all_event_urls, num_events=None):
    '''Takes in list of URLs to past events. Returns 3 lists: urls, winners, methods, each representing a UFC fight.
       Set num_events to be the number of events to get fights from. Set to None if want all.'''
    if num_events is None:
        num_events = len(all_event_urls)
    
    all_fight_urls, all_winners, all_methods = [], [], []
    for i, event_url in enumerate(tqdm(all_event_urls[:num_events])):
        # For each event, get the fight urls and winners
        fight_urls, winners, methods = get_all_fights_in_event(event_url, get_results=True)
        all_fight_urls.extend(fight_urls)
        all_winners.extend(winners)
        all_methods.extend(methods)
    return all_fight_urls, all_winners, all_methods


# + colab={"base_uri": "https://localhost:8080/"} id="Q66lyNtAF-Vo" outputId="c7eb935e-443a-4498-8690-f09c8e8be3ab"
FIGHT_URLS, WINNERS, METHODS = get_all_fights(ALL_EVENT_URLS, num_events=NUM_EVENTS)
print(f"Got {len(FIGHT_URLS)} fights")
print(FIGHT_URLS)
print(WINNERS)
print(METHODS)

assert(len(FIGHT_URLS) == len(WINNERS))
assert(len(FIGHT_URLS) == len(METHODS))


# + [markdown] id="CzlsyBU6DdRE"
# ## Get fight tables
#

# + id="zJjLUhEyDcSs"
def get_labeled_fight_tables(fight_url):
    '''Convert fight url to dictionary of pandas tables of information.
       Before, gave a list of tables that was hard to understand.
       Now have Totals, Per Round Totals, Significant Strikes, Per Round Significant Strikes'''
    fight_tables = pd.read_html(fight_url)
    
    labeled_fight_tables = {}
    labeled_fight_tables['Totals'] = fight_tables[0]
    labeled_fight_tables['Per Round Totals'] = fight_tables[1]
    labeled_fight_tables['Significant Strikes'] = fight_tables[2]
    labeled_fight_tables['Per Round Significant Strikes'] = fight_tables[3]
    return labeled_fight_tables


# + id="08jcNbZaDlBE"
RAW_FIGHT_TABLES_LIST = []
for url in tqdm(FIGHT_URLS):
    RAW_FIGHT_TABLES_LIST.append(get_labeled_fight_tables(url))

# + id="c9msProI12dH"
RAW_FIGHT_TABLES_LIST[0]['Totals'].head()

# + id="5_IIeQRx13WJ"
RAW_FIGHT_TABLES_LIST[0]['Per Round Totals'].head()

# + id="b8vW4zw818TK"
RAW_FIGHT_TABLES_LIST[0]['Significant Strikes'].head()

# + id="LtCciS5g16MB"
RAW_FIGHT_TABLES_LIST[0]['Per Round Significant Strikes'].head()


# + [markdown] id="r6YwJd-fAOwd"
# ## Clean fight information
#
# Separate each fighter's information into a different column
#
# TODO: Lots of stuff to improve. Smarter use of Totals, round by round, and significant strikes. Can also use non integer information, total attempted strikes (not just landed), fighter information, etc. All of those being ignored right now. Find nice way to parse new information round by round. Handle no winner case better. May need to add ignore_index=True for pd.concat

# + id="-PfTg13LB3ck"
def parse_string(row_string):
    '''Break string into two parts: one for fighter 0 and one for fighter 1
       Eg. 150 of 284  62 of 209 => (150 of 284, 62 of 209)'''
    if not isinstance(row_string, str):
        return "0", "0"
    string_split = row_string.split(" ")
    first_fighter_stat = " ".join(string_split[:len(string_split)//2])
    second_fighter_stat = " ".join(string_split[len(string_split)//2+1:])
    return first_fighter_stat, second_fighter_stat


# + id="dqnRE1IfMY9k"
def convert_to_int_or_double_if_possible(string):
    '''Convert string to int or double if possible
       If has a percent sign, tries to remove it and continue.'''
    def isfloat(value):
        try:
            float(value)
            return True
        except ValueError:
            return False

    # If input is not string, then return it unchanged
    if not isinstance(string, str):
        return string

    # Remove %
    if "%" in string:
        string = string.strip("%")

    # Convert to int or float
    if isfloat(string) and float(string).is_integer():
        return int(string)
    if isfloat(string):
        return float(string)
    return string


# + id="ORlZYocyRO4M"
def process_fight(raw_fight_table):
    '''Takes in a raw, one-row pandas fight table. Returns a pandas dataframe representing the fight statistics'''    
    # Break up columns.
    # Eg. "Name" => "Fighter 0 Name", "Fighter 1 Name"
    # "KD" => "Fighter 0 KD", "Fighter 1 KD"
    new_columns = []
    for column in raw_fight_table.columns:
        new_columns.append(f"Fighter 0 {column}")
        new_columns.append(f"Fighter 1 {column}")

    # Go through each row and break up the data into the columns
    new_rows = []
    for i, row in raw_fight_table.iterrows():
        new_row = []
        for column in raw_fight_table.columns:
            # Split string at the center space
            stat1, stat2 = parse_string(row[column])

            # TODO: Update this to capture more information

            # Has "100 of 120" type stat. Just store first number
            if " of " in stat1:
                stat1 = stat1.split(" of ")[0]
            if " of " in stat2:
                stat2 = stat2.split(" of ")[0]

            # Has "2:32" type stat (min:sec). Convert to sec.
            if len(re.findall("^[0-9]+:[0-9]+$", stat1)) > 0:
                min1, sec1 = stat1.split(":")[0], stat1.split(":")[1]
                stat1 = convert_to_int_or_double_if_possible(min1)*60 + convert_to_int_or_double_if_possible(sec1)
            if len(re.findall("^[0-9]+:[0-9]+$", stat2)) > 0:
                min2, sec2 = stat2.split(":")[0], stat2.split(":")[1]
                stat2 = convert_to_int_or_double_if_possible(min2)*60 + convert_to_int_or_double_if_possible(sec2)
            
            # Convert string to float or int if possible
            stat1 = convert_to_int_or_double_if_possible(stat1)
            stat2 = convert_to_int_or_double_if_possible(stat2)

            # Add to row
            new_row.append(stat1)
            new_row.append(stat2)
        new_rows.append(new_row)

    # Bring together into new dataframe, then only store the numerical values
    # TODO: Process better to keep more info, not throw so much away
    df = pd.DataFrame(new_rows, columns=new_columns)

    # Add in names, using smarter parsing
    df = df.drop(columns=['Fighter 0 Fighter', 'Fighter 1 Fighter'])
    fighters_string = raw_fight_table["Fighter"][0]  # Only 1 row table
    fighter0, fighter1 = get_fighters(fighters_string)
    df['Fighter 0 Name'] = fighter0
    df['Fighter 1 Name'] = fighter1
    return df


# + id="5oIdF2niZpag"
def process_raw_fight_tables(raw_fight_tables, winner, method, round_by_round=False):
    '''Takes in set of raw fight table (one fight), the name of the fight winner, and the method of winning. Returns a cleaned pandas table.
       Set round_by_round=True to use the round-by-round data. Otherwise, uses full fight stats.'''
    def create_aggregated_fight_table(raw_fight_tables):
        # Aggregate data from multiple tables
        fight_table = process_fight(raw_fight_tables["Totals"])
        fight_table2 = process_fight(raw_fight_tables["Significant Strikes"])
        
        # Rename column names with identical data to match
        fight_table2 = fight_table2.rename(columns={"Fighter 0 Sig. str": "Fighter 0 Sig. str.", "Fighter 1 Sig. str": "Fighter 1 Sig. str."})

        # Bring tables together, then remove duplicates
        fight_table = pd.concat([fight_table, fight_table2], axis=1)
        fight_table = fight_table.loc[:,~fight_table.columns.duplicated()]
        return fight_table

    def create_aggregated_round_by_round_fight_table(raw_fight_tables):
        ##### Aggregate data totals table
        tables = []
        for i, row in raw_fight_tables["Per Round Totals"].iterrows():
            # Get df of one round
            df = pd.DataFrame(row)
            values = list(df[i].to_dict().values())
            cols = list(raw_fight_tables["Totals"].columns)
            df = pd.DataFrame([values], columns=cols)

            # Update columns with round number
            new_cols = [f"Round {i+1} {c}" if c != "Fighter" else c for c in cols]
            df.columns = new_cols
            tables.append(process_fight(df))
        # Concatenate round-by-round horizontally, so each row is for 1 fight.
        # Then remove duplicates
        totals_df = pd.concat(tables, axis=1)
        totals_df = totals_df.loc[:,~totals_df.columns.duplicated()]

        ##### Aggregate data significant strikes table
        tables = []
        for i, row in raw_fight_tables["Per Round Significant Strikes"].iterrows():
            # Get df of one round
            df = pd.DataFrame(row)
            values = list(df[i].to_dict().values())
            cols = list(raw_fight_tables["Significant Strikes"].columns)
            if len(values) != len(cols):
                values = values[:-1]  # Remove last column values, as shown above, has extra column for no reason
            df = pd.DataFrame([values], columns=cols)

            # Update columns with round number
            new_cols = [f"Round {i+1} {c}" if c != "Fighter" else c for c in cols]
            df.columns = new_cols
            tables.append(process_fight(df))
        # Concatenate round-by-round horizontally, so each row is for 1 fight
        # Then remove duplicates
        sig_strikes_df = pd.concat(tables, axis=1)
        sig_strikes_df = sig_strikes_df.loc[:,~sig_strikes_df.columns.duplicated()]
        
        ##### Bring tables together, then remove duplicates
        fight_table = pd.concat([totals_df, sig_strikes_df], axis=1)
        fight_table = fight_table.loc[:,~fight_table.columns.duplicated()]
        return fight_table


    if round_by_round:
        fight_table = create_aggregated_round_by_round_fight_table(raw_fight_tables)
    else:
        fight_table = create_aggregated_fight_table(raw_fight_tables)

    if fight_table["Fighter 0 Name"][0] == winner:
        label = 0
    elif fight_table["Fighter 1 Name"][0] == winner:
        label = 1
    else:
        print(f'ERROR: fight_table["Fighter 0 Name"]={fight_table["Fighter 0 Name"]}, fight_table["Fighter 1 Name"]={fight_table["Fighter 1 Name"]}, winner={winner}')
        label = -1
    fight_table['Winner'] = label
    fight_table['Method'] = method
    return fight_table

# + id="BUyy5MUhNTkJ"
FIGHT_TABLE = []
for i in tqdm(range(len(RAW_FIGHT_TABLES_LIST))):
    FIGHT_TABLE.append(process_raw_fight_tables(RAW_FIGHT_TABLES_LIST[i], WINNERS[i], METHODS[i], round_by_round=ROUND_BY_ROUND)) 
FIGHT_TABLE = pd.concat(FIGHT_TABLE, ignore_index=True)
FIGHT_TABLE = FIGHT_TABLE.replace("^-+", np.nan, regex=True)  # Replace -- and --- with nan

# + id="G9EhqLLcAWs-"
FIGHT_TABLE.head()

# + id="7hQjO9B2RDoZ"
FIGHT_TABLE.tail()


# + [markdown] id="pCMOvzM0efI4"
# ## Augment dataset by flipping around columns
#
# The system should work the same no matter what order we pass in the fighters. Let fighters be A and B. We want
#
# winner(fighter0=A, fighter1=B) = winner(fighter0=B, fighter1=A)

# + id="kM2b_cAif7rM"
def create_flipped_table(table):
    '''Rearranges columns of table so that each fight has two rows. Let fighters be A and B.
       One row has (Fighter 0 = A, Fighter 1 = B). One row has (Fighter 0 = B, Fighter 1 = A)
       Ensure same column order, as column names not looked at when passed to ML model'''

    # Get columns in flipped order, which moves the columns around, but changes column name order too
    flipped_columns = []
    for column in table.columns:
        if "Fighter 0" in column:
            flipped_columns.append(column.replace("Fighter 0", "Fighter 1"))
        elif "Fighter 1" in column:
            flipped_columns.append(column.replace("Fighter 1", "Fighter 0"))
        else:
            flipped_columns.append(column)
    flipped_table = table[flipped_columns]

    # Flips winners around
    if 'Winner' in flipped_table.columns:
         flipped_table['Winner'] = flipped_table['Winner'].replace([0, 1], [1, 0])

    # Change column names back to normal
    flipped_table.columns = table.columns
    return flipped_table


# + id="KQcGgKW6k-ba"
def add_rows_of_flipped_columns(table):
    flipped_table = create_flipped_table(table)
    new_table = pd.concat([table, flipped_table])
    return new_table


# + id="HnwZdNiplLF3"
FULL_FIGHT_TABLE = add_rows_of_flipped_columns(FIGHT_TABLE)

# + id="PlnOp-fbjknE"
FULL_FIGHT_TABLE.head()

# + [markdown] id="gu7-RmZOkP68"
# ## Example of augmented data

# + id="PHsGqr0_joHn"
FULL_FIGHT_TABLE[(FULL_FIGHT_TABLE['Fighter 0 Name'] == "Robert Whittaker") & (FULL_FIGHT_TABLE['Fighter 1 Name'] == "Kelvin Gastelum")]

# + id="samSx7Olj3vQ"
FULL_FIGHT_TABLE[(FULL_FIGHT_TABLE['Fighter 1 Name'] == "Robert Whittaker") & (FULL_FIGHT_TABLE['Fighter 0 Name'] == "Kelvin Gastelum")]

# + [markdown] id="3OOgguk84RJl"
# ## Additional data cleaning
#
# TODO: See if something better than replacing nan with 0. See if something better for labels than 0 and 1. Could remove fights with no winner, or handle them differently. Could remove fights that don't go to decision by removing based on Method.

# + id="RIS0yarnbTmj"
X = FIGHT_TABLE.drop(['Winner', 'Fighter 0 Name', 'Fighter 1 Name', 'Method'], axis=1).fillna(0)
y = FIGHT_TABLE[['Winner']]

# + id="QxOiDLXHfgDx"
X.head()

# + id="N5qqnw6Efh8K"
y.head()

# + [markdown] id="JwvrHfOCf1mh"
# ## Setup train/validate/test split
# Can't blindly use full fight table train/validate/test split, because the augmented data must stay together. If in train we know winner(A, B) = A, then we don't want to have winner(B, A) in the validation/test set.

# + id="CwlwAWNRcwJ1"
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=0)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.33, random_state=0)
X_train, y_train = add_rows_of_flipped_columns(X_train), add_rows_of_flipped_columns(y_train)
X_valid, y_valid = add_rows_of_flipped_columns(X_valid), add_rows_of_flipped_columns(y_valid)
X_test, y_test = add_rows_of_flipped_columns(X_test), add_rows_of_flipped_columns(y_test)

# + id="aFmWIOydoJXd"
# Expect equal number of examples in Fighter 0 as Fighter 1
assert(len(y_train[y_train['Winner'] == 0]) == len(y_train[y_train['Winner'] == 1]))
assert(len(y_valid[y_valid['Winner'] == 0]) == len(y_valid[y_valid['Winner'] == 1]))
assert(len(y_test[y_test['Winner'] == 0]) == len(y_test[y_test['Winner'] == 1]))

# + id="jekwTdNAk3rE"
X_train.head()

# + id="BUeeqFtHpZQw"
y_train.head()

# + id="75PnIBkYpabr"
print(f"X_train.shape = {X_train.shape}")
print(f"X_valid.shape = {X_valid.shape}")
print(f"X_test.shape = {X_test.shape}")
print(f"y_train.shape = {y_train.shape}")
print(f"y_valid.shape = {y_valid.shape}")
print(f"y_test.shape = {y_test.shape}")

# + [markdown] id="ARUH8kxCbJpG"
# ## ML Models

# + id="0_v4cnEFbKp3"
from sklearn.ensemble import RandomForestClassifier

# + id="6gOrDS8AbPqM"
# Train
clf = RandomForestClassifier(max_depth=5, random_state=0)
clf.fit(X_train, y_train)

# Validate
accuracy_train = clf.score(X_train, y_train)
accuracy_valid = clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + id="dn1Njq7ecfAT"
import matplotlib.pyplot as plt

# Visualize importances
plt.rcParams.update({'font.size': 8})
plt.barh(X_train.columns, clf.feature_importances_)

# + id="GifEEZiTq2yL"
# MLP
from sklearn.neural_network import MLPClassifier

clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)
accuracy_train = clf.score(X_train, y_train)
accuracy_valid = clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + id="r6tiCNo3rEE0"
# SVM
from sklearn.svm import SVC

clf = SVC(random_state=1).fit(X_train, y_train)
accuracy_train = clf.score(X_train, y_train)
accuracy_valid = clf.score(X_valid, y_valid)
print(f"accuracy_train = {accuracy_train}")
print(f"accuracy_valid = {accuracy_valid}")

# + id="KNxPPw2DrbpW"
# FFN
import tensorflow as tf

model = tf.keras.models.Sequential()
model.add(tf.keras.Input(shape=X_train.shape[1:]))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(32, activation='relu'))
model.add(tf.keras.layers.Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
tf.keras.utils.plot_model(model, show_shapes=True, rankdir="LR")

# + id="agRwGSv2IEKa"
model.summary()

# + id="Wo9gE7_HtQhl"
model.fit(X_train, y_train, epochs=100, validation_data=(X_valid, y_valid))

# + id="RWTGwJUVtalk"
model.evaluate(X_train, y_train)
model.evaluate(X_valid, y_valid)

# + [markdown] id="ZOwm0hZTxZyr"
# ## Test out model manually

# + id="OWIypYX-uryi"
idx = 6

# + id="ofKgNtPPuC0V"
X_test.iloc[idx]

# + id="4tEAEW59ulsz"
# 0 means fighter 0 won. 1 means fighter 1 won.
y_test.iloc[idx]

# + id="67EbW0E1uXGi"
X_test.shape

# + id="3FWZP5LfuYYJ"
X_test.iloc[idx].shape

# + id="W19rlXXouGfs"
model.predict(np.expand_dims(X_test.iloc[idx], 0))

# + [markdown] id="ZOwm0hZTxZyr"
# ## Save data
#
# Store beginning file parameters.
# Use current date and time to save files uniquely.

# +
from datetime import datetime

now = datetime.now()
dt_string = now.strftime("%d-%m-%Y_%H:%M:%S")
print("dt_string =", dt_string)	
# -

parameters_string = f"NUM_EVENTS_{NUM_EVENTS_INPUT}_DATA_MODE_{DATA_MODE_INPUT}"
print("parameters_string =", parameters_string)	

import pickle
filename1 = f"FULL_FIGHT_TABLE_{parameters_string}_{dt_string}.csv"
filename2 = f"FIGHT_TABLE_{parameters_string}_{dt_string}.csv"
filename3 = f"ALL_FIGHTERS_{parameters_string}_{dt_string}.csv"
filename4 = f"RAW_FIGHT_TABLES_LIST_{parameters_string}_{dt_string}.pkl"
print(f"Saving to {filename1} and {filename2} and {filename3} and {filename4}")
FULL_FIGHT_TABLE.to_csv(filename1, index=False)
FIGHT_TABLE.to_csv(filename2, index=False)
ALL_FIGHTERS.to_csv(filename3, index=False)
with open(filename4, 'wb') as handle:
    pickle.dump(RAW_FIGHT_TABLES_LIST, handle, protocol=pickle.HIGHEST_PROTOCOL)

new = pd.read_csv(filename1)

new

with open(filename4, 'rb') as pickle_file:
    new2 = pickle.load(pickle_file)

len(new2[0])


# ## Experimental: Get detailed fighter information
#
# TODO: Get more detailed information about fighters, so we can change the task to fight prediction using fighter stats only. http://ufcstats.com/statistics/fighters?char=a&page=all has little information compared to http://ufcstats.com/fighter-details/33a331684283900f. Still lots to improve. Better features like strikes per minute. Handling nans better. Handling non win/losses better.

def get_all_fighters_detailed():
    '''Get pandas table with detailed information about all UFC fighters (KO's, strikes, etc.)'''
    fighter_detailed_tables = []
    
    # For each letter of the alphabet, get the fighters
    for c in tqdm(ascii_lowercase):
        # Each page has a list of fighter detail urls
        all_fighters_url = f"http://ufcstats.com/statistics/fighters?char={c}&page=all"
        all_fighters_html = urlopen(all_fighters_url).read().decode("utf-8")

        # Regex for "http://ufcstats.com/fighter-details/<alphanumeric>"
        # Eg. "http://ufcstats.com/fighter-details/27541033b97c076d"
        pattern = "\"http://ufcstats.com/fighter-details/[a-zA-Z0-9_]+\""
        urls = re.findall(pattern, all_fighters_html)
        
        # Remove quotes and duplicates
        urls = [url.strip("\"") for url in urls]
        urls = remove_duplicates_keep_order(urls)
        
        # For each fighter detail url, merge together their record information
        # Initially in form "Eddie Alvarez Rafael Dos Anjos", "0 0", "1:10, 0:00"
        # Want just "Eddie Alvarez", "0", "1:10", then convert to numbers
        # Just need to get the first value of each one, then average/sum/aggregate this together
        for url in urls:
            fighter_table = pd.read_html(url)[0].dropna(subset=["Time"], how='all')  # Drop initial row of nans

            # If no fight information, add empty dataframe
            if fighter_table.shape[0] == 0:
                df = pd.DataFrame()
                fighter_detailed_tables.append(df)
                continue
                
            # Preprocess certain values for consistency
            # TODO: Handle this better, perhaps keep more information
            fighter_table = fighter_table.drop(columns=["Method", "Event"])
            fighter_table.loc[~fighter_table['W/L'].isin(['win', 'loss']), 'W/L'] = "-1 -1"
            fighter_table.loc[fighter_table['W/L'] == 'win', 'W/L'] = "1  1"
            fighter_table.loc[fighter_table['W/L'] == 'loss', 'W/L'] = "0  0"
            times = [int(min_) * 60 + int(sec) for min_, sec in fighter_table['Time'].str.split(':')]
            fighter_table['Time'] = [f"{t}  {t}" for t in times]
            
            # Parse each row to remove the other fighter's information
            new_rows = []
            for i, row in fighter_table.iterrows():
                # Get df of one round
                df = pd.DataFrame(row, columns=fighter_table.columns)
                values = [row[col] for col in df.columns]
                df = pd.DataFrame([values], columns=fighter_table.columns)
                df = process_fight(df)
                new_rows.append(df)

            # Put rows together, then only keep Fighter 0, then remove "Fighter 0 "
            totals_df = pd.concat(new_rows)
            totals_df = totals_df.loc[:, totals_df.columns.str.contains('Fighter 0')]
            totals_df.columns = [col.replace("Fighter 0 ", "") for col in totals_df.columns]
            totals_df = totals_df.replace("^-+", np.nan, regex=True)  # Replace -- and --- with nan

            # Summarize fighter in 1 row
            new_columns = []
            new_row = []
            for col in totals_df.columns:
                if col == "Name":
                    new_columns.append(col)
                    new_row.append(totals_df[col].iloc[0])
                else:
                    total_col = f"{col} Total"
                    avg_col = f"{col} Avg"
                    new_columns.extend([total_col, avg_col])
                    total = totals_df[col].sum()
                    avg = totals_df[col].mean()
                    new_row.extend([total, avg])
            totals_df = pd.DataFrame([new_row], columns=new_columns)

            fighter_detailed_tables.append(totals_df) 
            break  # Remove this when ready
    all_fighters = pd.concat(fighter_detailed_tables)
    return all_fighters


x = get_all_fighters_detailed()

x.head()




