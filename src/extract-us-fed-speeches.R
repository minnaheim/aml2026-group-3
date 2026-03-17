
library(tidyverse)

# us_speeches <- read.csv(".../data/us_speeches.csv")[, -1]
us_speeches <- read.csv("data/us_speeches.csv")[, -1]
# print(names(us_speeches))
# sort by dates
us_speeches <- us_speeches |> arrange(Date)
## Data Cleaning

### Pruning Text
# each datasource differently introduces the speaker 
# -> i think this is not necessary for sentiment

# let's see how often each data source appears

us_speeches |> count(Source, sort = TRUE)



#### For CB websites


cb_test <- us_speeches |> filter(Source == "CB websites") |> select(Date, text) |> slice_head(n=6)
# View(cb_test)


# remove this first part from all CB websites speeches

library(lubridate)

# test cleaning

# Assuming your dataframe is called 'df' and columns are 'date' and 'speech'
cb_speeches_cleaned <- us_speeches %>%
  filter(Source == "CB websites") %>%
  mutate(
    # 1. Parse the date column (just in case it's currently a character)
    parsed_date = ymd(Date),
    
    # 2. Reconstruct the date as it appears in the text. 
    # We use day() instead of standard formatting to avoid leading zeros (e.g., "May 4" not "May 04")
    text_date = paste0(
      month(parsed_date, label = TRUE, abbr = FALSE, locale = "en_US.UTF-8"), " ", 
      day(parsed_date), ", ", 
      year(parsed_date)
    ),
    
    # 3. Create a dynamic regex pattern for each row.
    # (?i) makes it case-insensitive.
    # (?s) allows the dot (.) to match across newlines if your text has line breaks.
    # .*? is non-greedy, so it stops at the FIRST instance of the date.
    pattern = paste0("(?i)(?s).*?", text_date),
    
    # 4. Remove the matched text and trim any leftover leading punctuation or spaces
    cleaned_speech = str_remove(text, pattern) %>% 
      str_trim() %>% 
      # Optional: Remove leading punctuation like dashes or commas left behind
      str_remove("^[,\\-\\s]+") 
  )  %>%
  
  # Drop the intermediate columns to keep it tidy
  select(-parsed_date, -text_date, -pattern)

#as we can see, the speaker starts speaking after the first time the date is mentioned, in this case 

# remove the initial introduction of speaker, aka everything until *** -> in BIS data

# but first, let's see if this appears everywhere

