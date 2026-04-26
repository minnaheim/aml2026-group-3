# in case its not installed 
# install.packages("fredr")
library(fredr)

# could just use the API too.
series <- c("CPIAUCSL", "payems", "indpro", "unrate")
key <- "182161f35ab1b0231ab7a21e3b991a52"
fredr_set_key(key)
# metadata <- fredr_series(series_id = "UNRATE")

# cannot send multiple keys

df <- data.frame()

for (el in series){
  meta <- fredr_series(series_id = el)
  df <- rbind(df, meta)
}

View(df)
write.csv(df, "data/metadata-macro-monthly.csv")
