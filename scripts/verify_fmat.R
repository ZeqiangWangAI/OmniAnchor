# Independent base-R check of the Python stored-score correlation calculation.
args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 2, !file.exists(args[2]))
rows <- list()
for (study in c("d1a", "d1b")) {
  d <- read.csv(file.path(args[1], paste0(study, ".csv")))
  for (model in unique(d$model)) {
    subset <- d[d$model == model, ]
    male <- subset[subset$MASK == "Male", c("query", "T_word", "M_pair", "prob")]
    female <- subset[subset$MASK == "Female", c("query", "T_word", "M_pair", "prob")]
    paired <- merge(male, female, by=c("query", "T_word", "M_pair"), suffixes=c(".male", ".female"))
    paired$lpr <- log(paired$prob.male) - log(paired$prob.female)
    paired$z <- ave(paired$lpr, paired$query, FUN=function(x) as.numeric(scale(x)))
    if (study == "d1a") {
      paired$T_word <- sub("^(a|an) ", "", paired$T_word)
    }
    target <- aggregate(z ~ T_word, paired, mean)
    if (study == "d1a") {
      gold <- read.csv(file.path(args[1], "stats.occupation.csv"))
      merged <- merge(target, gold, by.x="T_word", by.y="job")
    } else {
      gold <- read.csv(file.path(args[1], "stats.name.csv"))
      merged <- merge(target, gold, by.x="T_word", by.y="name")
    }
    rows[[length(rows) + 1]] <- data.frame(study=study, model=model,
      pearson_r=cor(merged$z, merged$P_male), spearman_rho=cor(merged$z, merged$P_male, method="spearman"))
  }
}
write.csv(do.call(rbind, rows), args[2], row.names=FALSE)
