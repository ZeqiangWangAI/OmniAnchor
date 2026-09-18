args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args) == 2)
out <- args[[1]]
source <- args[[2]]
d <- read.csv(file.path(out, "template-scores.csv"), check.names=FALSE)
z <- ave(d$lpr, interaction(d$study, d$method, d$template, drop=TRUE),
         FUN=function(x) (x-mean(x))/sd(x))
stopifnot(max(abs(z-d$z)) < 1e-12)
pred <- aggregate(z, list(study=d$study, method=d$method, target=d$target), mean)
names(pred)[[4]] <- "z"
reported <- read.csv(file.path(out, "matched-correlations.csv"), check.names=FALSE)
checks <- list()
for (study in c("d1a", "d1b")) {
  file <- if (study == "d1a") "stats.occupation.csv" else "stats.name.csv"
  key <- if (study == "d1a") "job" else "name"
  gold <- read.csv(file.path(source, file), check.names=FALSE)
  names(gold)[names(gold) == key] <- "target"
  for (method in unique(pred$method)) {
    p <- merge(pred[pred$study == study & pred$method == method, ], gold, by="target")
    stopifnot(nrow(p) == 50)
    r <- cor(p$z, p$P_male, method="pearson")
    rho <- cor(p$z, p$P_male, method="spearman")
    expected <- reported[reported$study == study & reported$method == method & reported$subset == "all50", ]
    stopifnot(nrow(expected) == 1, abs(r-expected$pearson_r) < 1e-12,
              abs(rho-expected$spearman_rho) < 1e-12)
    checks[[length(checks)+1]] <- data.frame(study=study, method=method, pearson=r,
      spearman=rho, max_error=max(abs(r-expected$pearson_r), abs(rho-expected$spearman_rho)))
  }
}
write.csv(do.call(rbind, checks), file.path(out, "independent-R-verification.csv"), row.names=FALSE)
capture.output(sessionInfo(), file=file.path(out, "R-session.txt"))
cat("Independent R normalization, gold joins, Pearson and Spearman all verified within1e-12.\n")
