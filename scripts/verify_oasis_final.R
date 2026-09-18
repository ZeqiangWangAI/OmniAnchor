args <- commandArgs(trailingOnly=TRUE)
stopifnot(length(args)==3)
folder <- args[1]
labels <- read.csv(args[2], check.names=FALSE)
reported <- read.csv(args[3], check.names=FALSE)
raw <- read.csv(file.path(folder, "native-raw.csv"), check.names=FALSE)
stopifnot(!anyDuplicated(raw[c("sample_id", "anchor_id", "bridge_id")]))
avg <- aggregate(raw_logp ~ sample_id + anchor_id, data=raw, FUN=mean)
native <- reshape(avg, idvar="sample_id", timevar="anchor_id", direction="wide")
names(native) <- sub("^raw_logp\\.", "", names(native))
inputs <- list(OmniAnchor=native,
  `qwen-embedding`=read.csv(file.path(folder,"qwen-embedding.csv"), check.names=FALSE),
  `qwen-reranker`=read.csv(file.path(folder,"qwen-reranker.csv"), check.names=FALSE))
checks <- list()
for (method in names(inputs)) {
  x <- inputs[[method]]
  stopifnot(!anyDuplicated(x$sample_id), setequal(x$sample_id, labels$sample_id), nrow(x)==201)
  x <- x[match(labels$sample_id, x$sample_id),]
  contrasts <- list(V=x$pleasant-x$unpleasant, A=x$aroused-x$calm)
  for (dimension in names(contrasts)) {
    for (metric in c("Pearson", "Spearman")) {
      value <- cor(contrasts[[dimension]], labels[[dimension]], method=tolower(metric))
      row <- reported[reported$method==method & reported$dimension==dimension & reported$metric==metric,]
      stopifnot(nrow(row)==1, abs(value-row$correlation)<1e-12)
      checks[[length(checks)+1]] <- data.frame(method=method, dimension=dimension,
        metric=metric, independently_computed=value, absolute_error=abs(value-row$correlation))
    }
  }
}
write.csv(do.call(rbind,checks),file.path(folder,"verification.csv"),row.names=FALSE)
capture.output(sessionInfo(),file=file.path(folder,"R-session.txt"))
cat("12 final correlations verified from raw bridges and original gold joins\n")
