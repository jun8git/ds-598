raw_data_with_labels <- read_lines("raw.en")
raw_data_without_labels <- raw_data_with_labels %>%
  str_replace_all(c("<eob>"="", "<eol>"="")) %>%
  str_squish()

n <- length(raw_data_with_labels)

n_trn <- as.integer(n * 0.8)
n_tst <- as.integer(n * 0.1)
n_val <- n - (n_trn + n_tst)

train_with_labels <- raw_data_with_labels[1:n_trn]
test_with_labels <- raw_data_with_labels[(n_trn + 1):(n_trn + n_tst)]
val_with_labels <- raw_data_with_labels[(n_trn+n_tst+1):n]

train_without_labels <- raw_data_without_labels[1:n_trn]
test_without_labels <- raw_data_without_labels[(n_trn + 1):(n_trn + n_tst)]
val_without_labels <- raw_data_without_labels[(n_trn+n_tst+1):n]

train_with_labels %>% write_lines("train_with_labels.txt")
test_with_labels %>% write_lines("test_with_labels.txt")
val_with_labels %>% write_lines("val_with_labels.txt")

train_without_labels %>% write_lines("train_without_labels.txt")
test_without_labels %>% write_lines("test_without_labels.txt")
val_without_labels %>% write_lines("val_without_labels.txt")
