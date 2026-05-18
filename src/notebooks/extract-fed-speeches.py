#!/usr/bin/env python
# coding: utf-8

# ## Cleaning the Fed Speech data

# set-up
import pandas as pd
import os 
from rapidfuzz import process, fuzz


# print(os.getcwd())
speeches = pd.read_csv("../data/us_speeches.csv")

# speeches
# type(speeches)

# ### first, check average word count as is
# Then, clean each of the different sources of their headers, intros, etc. to get just the raw speech content

speeches["word_count"] = speeches["text"].apply(lambda x: len(x.split()))
speeches.word_count.mean()

speeches.loc[speeches["Source"] == "CB websites"].word_count.mean()

speeches.Source.value_counts()

# ### Now we clean the "CB websites" first

# select only cb source speeches
cb = speeches.loc[speeches["Source"] == "CB websites"]

# create a parsed date variable which includes the date in the format "Month Day, Year" for each row
# but in case the date is 2008-01-08, it should be converted to "January 8, 2008", not "January 08, 2008"
cb["parsed_date"] = cb.apply(lambda row: pd.to_datetime(row["Date"]).strftime("%B %-d, %Y"), axis=1)

# create a new column text-cleaned which includes only the text after the parsed date is mentioned in the text column for the first time
cb["text-cleaned"] = cb.apply(lambda row: row["text"].split(row["parsed_date"])[-1] if row["parsed_date"] in row["text"] else row["text"], axis=1)
# cb["text-cleaned"] = cb.apply(lambda row: row["text"].split(row["parsed_date"])[-1], axis=1)

# in case text stats with "Last Updated", remove everything before and incl. "Introduction"
cb["text-cleaned"] = cb.apply(lambda row: row["text-cleaned"].split("Introduction")[-1] if "Last Updated" in row["text-cleaned"] else row["text-cleaned"], axis=1)

# Ok, some of these datasets still are uncleaned, i.e. those that start with:
# 
# i.e. if it starts with "Last Updated" remove until "Introduction"

# ## remove end of speech 
# 
# patterns to match:
# -> "Thank you. NOTES:" 
# 


# remove the part of the text, after "Thank you. NOTES:"
cb["text-cleaned"] = cb.apply(lambda row: row["text-cleaned"].split("Thank you. NOTES:")[0] if "Thank you. NOTES:" in row["text-cleaned"] else row["text-cleaned"], axis=1)


# recalculate word count for the cleaned text
cb["word_count_cl"] = cb["text-cleaned"].apply(lambda x: len(x.split()))
cb["word_count_dirt"] = cb["text"].apply(lambda x: len(x.split()))

cb["word_count_cl"].mean()
cb["word_count_dirt"].mean()

# to the cb dataframe, add a column called "words-removed" which incl. the difference in word count between the original text and the cleaned text
cb["words-removed"] = cb["word_count_dirt"] - cb["word_count_cl"]
cb["words-removed"].head()

# if word_count_cl is less than 100, then replace text-cleaned with the original text
cb["text-cleaned"] = cb.apply(lambda row: row["text"] if row["word_count_cl"] < 100 else row["text-cleaned"], axis=1)
cb["word_count_cl"] = cb["text-cleaned"].apply(lambda x: len(x.split()))
cb["words-removed"] = cb["word_count_dirt"] - cb["word_count_cl"]


# ### Clean BIS data 
speeches.loc[speeches["Source"] == "BIS"].word_count.mean()

# select only cb source speeches
bis = speeches.loc[speeches["Source"] == "BIS"]

# bis has "* * *" in the text, remove everything before and incl. "* * *"
bis["text-cleaned"] = bis.apply(lambda row: row["text"].split("* * * ")[-1] if "* * * " in row["text"] else row["text"], axis=1)
bis["word_count_cl"] = bis["text-cleaned"].apply(lambda x: len(x.split()))
bis["word_count_dirt"] = bis["text"].apply(lambda x: len(x.split()))
bis["words-removed"] = bis["word_count_dirt"] - bis["word_count_cl"]
bis["words-removed"].head()
# bis.to_csv("../data/bis_speeches.csv", index=False)

# ### Clean Archive Dataset
# 


arch = speeches.loc[speeches["Source"] == "Archives"]

arch["parsed_date"] = arch.apply(lambda row: pd.to_datetime(row["Date"]).strftime("%B %-d, %Y"), axis=1)

# create a new column text-cleaned which includes only the text after the parsed date is mentioned in the text column for the first time
arch["text-cleaned"] = arch.apply(lambda row: row["text"].split(row["parsed_date"])[-1], axis=1)
# cb["text-cleaned"] = cb.apply(lambda row: row["text"].split(row["parsed_date"])[-1], axis=1)
arch["word_count_cl"] = arch["text-cleaned"].apply(lambda x: len(x.split()))
arch["word_count_dirt"] = arch["text"].apply(lambda x: len(x.split()))
arch["words-removed"] = arch["word_count_dirt"] - arch["word_count_cl"]
arch["words-removed"].head()
# arch.to_csv("../data/arch_speeches.csv", index=False)
arch



# if word_count_cl is less than 4, then replace text-cleaned with the original text
arch["text-cleaned"] = arch.apply(lambda row: row["text"] if row["word_count_cl"] < 4 else row["text-cleaned"], axis=1)
arch["word_count_cl"] = arch["text-cleaned"].apply(lambda x: len(x.split()))
arch["words-removed"] = arch["word_count_dirt"] - arch["word_count_cl"]
arch["words-removed"].head()
# arch.to_csv("../data/arch_speeches.csv", index=False)

# now reconcatentate all 3 dataframes into one
final_speeches = pd.concat([cb, bis, arch], ignore_index=True)
final_speeches.to_csv("../data/cleaned_speeches.csv", index=False)



# now, add the metadata of the speaker
# in metadata, have the "last_name" column and the "speaker_name" column with full name (as on fed history webpage)
# in speeches, have the "Authorname" column
# try fuzzy matching

# also, we have "term_start_precise" and "term_end_precise"
# e.g. John C Williams was FRB president in SF and NY, can match this appropriately!

# read data
speaker_metadata = pd.read_excel("../data/metadata_speakers.xlsx")


# normalize names: remove dots, strip whitespace, lowercase
final_speeches["last_name"] = final_speeches["Authorname"].str.replace(".", "", regex=False).str.strip().str.split().str[-1].str.lower()
speaker_metadata["last_name"] = speaker_metadata["speaker_name"].str.replace(".", "", regex=False).str.strip().str.split().str[-1].str.lower()

# date columns
final_speeches["Date"] = pd.to_datetime(final_speeches["Date"])
speaker_metadata["term_start_precise"] = pd.to_datetime(speaker_metadata["term_start_precise"])
speaker_metadata["term_end_precise"] = speaker_metadata["term_end_precise"].replace("end of sample", pd.NaT)
speaker_metadata["term_end_precise"] = pd.to_datetime(speaker_metadata["term_end_precise"])

# merge on last name, then filter by date range
merged = final_speeches.merge(speaker_metadata, on="last_name", how="left")

# only apply date filter where a match was found
# the data also contains speeches by senior staff 
# => we wan't to keep them, so filtering by term dates must incorporate this
matched_mask = merged["term_start_precise"].notna()

merged = merged[
    ~matched_mask |  # keep all unmatched rows (no metadata)
    (
        (merged["Date"] >= merged["term_start_precise"]) &
        (merged["Date"] <= merged["term_end_precise"].fillna(pd.Timestamp.max))
    )
]

print(merged.shape)
print(merged["speaker_name"].isna().sum())


merged.to_csv("../data/speeches_with_metadata.csv", index=False)

