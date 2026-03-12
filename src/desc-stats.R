library(tidyverse)

# only for me. i will not commit the original file, it's too large 
# but i will still show how i processed it

# source: https://cbspeeches.com/

# data <- readRDS("data/CBS_dataset_v1.0.rds")
# print(names(data))

# ### filter for country of interest 
# us_speeches <- data |> 
#   filter(Country == "USA") |>
#   mutate(
#     # Calculate word count for each speech
#     word_count = str_count(text, "\\S+")
#   ) |>
#   # remove text_original (all in english anywa)
#   select(-text_original)


# head(us_speeches)

# write.csv(us_speeches, file = "data/us_speeches.csv")

# FIRST UNZIP THE US SPEECHES, CANNOT COMMIT AS IS, TOO BIG
us_speeches <- read.csv("data/us_speeches.csv")[,-1]
print(names(us_speeches))

### --------------- summary statistics ----------------

# ---------- basic word count summy ----------
summary(us_speeches$word_count)
boxplot(us_speeches$word_count, outline = FALSE) # removed showing of outliers, not precise really

# ---------- speeches per year ----------
speeches_per_year <- us_speeches |>
  mutate(year = year(Date)) |>
  group_by(year) |>
  summarise(
    n_speeches = n(),
    avg_word_count = mean(word_count, na.rm = TRUE)
  ) |>
  arrange(year)

print(speeches_per_year)


# simple barplot 
# save barplot
png(filename = "out/speech_volume_barplot.png", width = 800, height = 600)
barplot(
  n_speeches ~ year, 
  data = speeches_per_year,
  xlab = "Year",
  ylab = "Number of Speeches",
  main = "Volume of US Central Bank Speeches Over Time",
  col = "skyblue"
)
dev.off()

# Visualization: Number of speeches per year
p_volume <- ggplot(speeches_per_year, aes(x = year, y = n_speeches)) +
  geom_line() +
  geom_point() +
  labs(title = "Volume of US Central Bank Speeches Over Time",
       x = "Year", y = "Number of Speeches") +
  theme_minimal()

# ---------- plot speeches (length) over time ----------
# Plot
p_avg_length <- ggplot(speeches_per_year, aes(x = year, y = avg_word_count)) +
  geom_line(color = "steelblue", size = 1) +
  geom_point() +
  labs(
    title = "Average Length of US Central Bank Speeches",
    subtitle = "Mean word count per year",
    x = "Year",
    y = "Average Word Count"
  ) +
  theme_minimal()

# barplot 
png(filename = "out/avg_length_barplot.png", width = 800, height = 600)
barplot(
  avg_word_count ~ year, data = speeches_per_year,
  xlab = "Year",
  ylab = "Avg. Word Count",
  main = "Average Length of US Central Bank Speeches",
  col = "blue"
)
dev.off()

# Average word count by Role
word_count_by_role <- us_speeches |>
  group_by(Role) |>
  summarise(
    mean_words = mean(word_count, na.rm = TRUE),
    median_words = median(word_count, na.rm = TRUE),
    n = n()
  ) |>
  filter(n > 5) |> # Filter out roles with very few speeches for better stats
  arrange(desc(mean_words))

print(word_count_by_role)


summary_stats <- us_speeches |>
  summarise(
    total_speeches = n(),
    unique_authors = n_distinct(Authorname),
    start_date = min(Date, na.rm = TRUE),
    end_date = max(Date, na.rm = TRUE),
    avg_words = mean(word_count, na.rm = TRUE),
    total_words = sum(word_count, na.rm = TRUE)
  )

print(summary_stats)


# write all these to out dir
ggsave("out/speech_volume.png", p_volume, width = 10, height = 6, dpi = 300)
ggsave("out/avg_length.png", p_avg_length, width = 10, height = 6, dpi = 300)

