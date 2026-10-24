# Figures 4 and 5: statistics.
# ART mixed models with mouse as a random intercept (ARTool):
#   Fig. 4, WT vs GtdKO+AAV-Gat2 pre-training:    value ~ group * level + (1 | mouse)       -> group
#   Fig. 5, WT vs post-training:                  value ~ group * level + (1 | mouse)       -> group
#   Fig. 5, pre vs post-training (same mice):     value ~ training * level + (1 | mouse)    -> training
#   Responsive fraction by SF (4C, 5A): ART-C contrasts at each SF, Benjamini-Hochberg over SFs.
#   Fig. 4C blue asterisks: pre-training fraction vs the 5% chance level, one-sided nonparametric bootstrap
#   over mice (100,000 resamples), Benjamini-Hochberg over SFs.
# The single-neuron tuning curves (4E, 5C) are descriptive and not tested.
suppressPackageStartupMessages(library(ARTool))
options(warn = -1, width = 140)
f <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
if (length(f)) setwd(dirname(normalizePath(f)))
set.seed(1)

d <- read.csv("fig45_session_metrics.csv", stringsAsFactors = FALSE)
res <- list()
add <- function(...) res[[length(res) + 1]] <<- data.frame(..., stringsAsFactors = FALSE)
term_p <- function(model_formula, x, term) {
  a <- anova(art(model_formula, data = droplevels(x)))
  a[a$Term == term, "Pr(>F)"]
}

COMPARISONS <- list(list(fig = "4", name = "WT vs Pre", groups = c("WT", "Pre")),
                    list(fig = "5", name = "WT vs Post", groups = c("WT", "Post")),
                    list(fig = "5", name = "Pre vs Post", groups = c("Pre", "Post")))

# ---------------------------------------------------------------- session-level main effects
for (metric in unique(d$metric)) for (cmp in COMPARISONS) {
  x <- d[d$metric == metric & d$group %in% cmp$groups, ]
  if (length(unique(x$group)) < 2) next
  x$level <- factor(x$level); x$mouse <- factor(x$mouse); x$grp <- factor(x$group, cmp$groups)
  p <- term_p(value ~ grp * level + (1 | mouse), x, "grp")     # pre/post share mouse IDs, so pairs are modelled
  add(figure = cmp$fig, metric = metric, comparison = cmp$name, level = NA, test = "ART group/training", p = p)
}

# ---------------------------------------------------------------- per-SF contrasts, responsive fraction
for (cmp in COMPARISONS) {
  x <- d[d$metric == "resp_by_sf" & d$group %in% cmp$groups, ]
  x$level <- factor(x$level); x$mouse <- factor(x$mouse); x$grp <- factor(x$group, cmp$groups)
  s <- as.data.frame(summary(art.con(art(value ~ grp * level + (1 | mouse), data = droplevels(x)), "grp:level",
                                     adjust = "none")))
  parts <- strsplit(as.character(s$contrast), " - ")
  same_level <- sapply(parts, function(p) sub("^[^,]+,", "", p[1]) == sub("^[^,]+,", "", p[2]))
  s <- s[same_level, ]
  lv <- sapply(strsplit(as.character(s$contrast), " - "), function(p) sub("^[^,]+,", "", p[1]))
  q <- p.adjust(s$p.value, "BH")
  for (i in seq_along(lv)) add(figure = cmp$fig, metric = "resp_by_sf", comparison = cmp$name, level = lv[i],
                               test = "ART-C, BH", p = q[i])
}

# ---------------------------------------------------------------- Fig. 4C: pre-training vs 5% chance
x <- d[d$metric == "resp_by_sf" & d$group == "Pre", ]
pb <- sapply(sort(unique(x$level)), function(lv) {
  v <- x$value[x$level == lv]
  boot <- colMeans(matrix(sample(v, 1e5 * length(v), replace = TRUE), nrow = length(v)))
  (sum(boot <= 0.05) + 1) / (length(boot) + 1)
})
qb <- p.adjust(pb, "BH")
for (i in seq_along(qb)) add(figure = "4", metric = "resp_by_sf", comparison = "Pre vs chance",
                             level = as.character(sort(unique(x$level))[i]), test = "bootstrap, BH", p = qb[i])

res <- do.call(rbind, res)
res$stars <- cut(res$p, c(-Inf, 1e-4, 1e-3, 1e-2, 0.05, Inf), c("****", "***", "**", "*", "n.s."))
write.csv(res, "fig45_stats.csv", row.names = FALSE)
print(transform(res, p = signif(p, 3)), row.names = FALSE)
