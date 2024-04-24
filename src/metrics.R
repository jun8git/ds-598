library(tidyverse)

get_f1_score <- function(pred_hds, hds){
    PP <- sum(pred_hds %in% c("eob", "eol"))
    AP <- sum(hds %in% c("eob", "eol"))
    TP <- sum(pred_hds[pred_hds == hds] %in% c("eob", "eol"))
    p <- TP/PP
    r <-TP/AP
    print(PP)
    print(AP)
    print(TP)
    print(p)
    print(r)
    return((2*p*r)/(p+r))
}

data <- read_csv(paste0(data_path, "/MustJ/exp/bert_token_hds_4_pred_val.csv"))
f1_score <- get_f1_score(data %>% pull(pred_hds), data %>% pull(hds))
print(f1_score)

compute_ngrams <- function(words, n) {
    num_ngrams <- length(words) - n + 1
    ngrams <- vector(mode = "list", length = num_ngrams)

    for (i in 1:num_ngrams) {
                                        #print(i)
                                        #print(num_ngrams)
        ngrams[[i]] <- paste(words[i:(i + n - 1)], collapse = " ")
    }

    return(ngrams)
}

                                        # Function to compute precision of n-grams
compute_precision <- function(candidate_ngrams, reference_ngrams) {
    count_correct <- sum(sapply(candidate_ngrams, function(ngram) ngram %in% reference_ngrams))
    precision <- count_correct / length(candidate_ngrams)
    return(precision)
}

                                        # Function to compute BLEU score
compute_bleu <- function(candidate_words, reference_words, max_ngram = 4) {
    max_ngram = min(max_ngram, length(candidate_words))
    bleu_scores <- numeric(max_ngram)

    for (n in 1:max_ngram) {
        candidate_ngrams <- compute_ngrams(candidate_words, n)
        reference_ngrams <- compute_ngrams(reference_words, n)

        precisions <- sapply(reference_ngrams, function(ref_ngram) {
            compute_precision(candidate_ngrams, ref_ngram)
        })

        bleu_scores[n] <- mean(precisions)
    }

    brevity_penalty <- min(1, exp(1 - (length(candidate_words) / length(reference_words))))
    bleu_score <- brevity_penalty * exp(mean(log(bleu_scores[bleu_scores > 0])))

    return(bleu_score)
}

bleu_score <- compute_bleu((data %>% pull(pred_hds))[1:10], (data %>% pull(hds))[1:10])

print(bleu_score)
