args <- commandArgs(trailingOnly=TRUE)
analysis <- args[1]; output <- args[2]; n <- as.integer(args[3])
expected <- read.csv(file.path(output, 'expected.csv'), check.names=FALSE)
gold <- readBin(file.path(output, 'gold.bin'), 'double', n=n, size=8, endian='little')
rows <- list()
for (i in seq_len(nrow(expected))) {
  method <- expected$method[i]
  pred <- readBin(file.path(output, paste0(method, '.bin')), 'double', n=n, size=8, endian='little')
  stopifnot(length(pred) == n, length(gold) == n)
  actual <- cor(gold, pred, method='spearman')
  rows[[length(rows)+1]] <- data.frame(method=method, metric='pooled_pair', expected=expected$spearman[i], actual=actual)
}
shift <- read.csv(file.path(analysis, 'per-target-shift.csv'), check.names=FALSE)
change <- read.csv(file.path(analysis, 'change-validity.csv'), check.names=FALSE)
for (i in seq_len(nrow(change))) {
  row <- change[i, ]; subset <- shift[shift$method == row$method, ]
  stopifnot(nrow(subset) == row$n_targets, !anyDuplicated(subset$target))
  actual <- cor(subset$human_change, subset[[row$metric]], method='spearman')
  rows[[length(rows)+1]] <- data.frame(method=row$method, metric=row$metric, expected=row$spearman, actual=actual)
}
result <- do.call(rbind, rows)
result$error <- abs(result$actual-result$expected)
write.csv(result, file.path(output, 'independent-correlations.csv'), row.names=FALSE)
stopifnot(all(is.finite(result$error)), max(result$error) < 1e-12)
cat(nrow(result), 'correlations verified; max error', max(result$error), '\n')
