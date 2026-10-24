# Training effect in WT and GtdKO+AAV-Gat2 mice (Supplementary Figure 3, Figure 6D).
#
# ART mixed models (ARTool), per metric (levels = SF, orientation or neuron number).
# Reported model (as for Figures 4-6), mouse as a random intercept, labelled "(1|mouse)" in the output:
#   within   : value ~ training * level + (1 | mouse), per group
#   between  : value ~ group * training * level + (1 | mouse); group x training term
#   post/pre : value ~ group * level + (1 | mouse) on post- or pre-training sessions
#   per-SF   : ART-C pre vs post at each SF for the responsive fraction (BH-FDR)
# Also fitted, as sensitivity analyses: the same models with a per-mouse random slope for training
# (1 + training | mouse); an exact permutation test of each mouse's change (post - pre, averaged over levels)
# between groups; VD029 included; WT restricted to mice imaged before VR training began (VD017, VD018, VD019).

suppressPackageStartupMessages(library(ARTool))
options(width = 140, warn = -1)
f <- sub("--file=", "", grep("--file=", commandArgs(FALSE), value = TRUE))
if (length(f)) setwd(dirname(normalizePath(f)))

d_all <- read.csv("session_metrics_long.csv", stringsAsFactors = FALSE)
LEVELS <- list(resp_by_sf = c(0.02, 0.04, 0.08, 0.16, 0.32, 0.64),
               resp_by_ori = c(0, 30, 60, 90, 120, 150),
               osi_by_sf = c(0.02, 0.04, 0.08, 0.16, 0.32),
               sf_dec_vs_n = seq(10, 80, 10), ori_dec_vs_n = seq(10, 80, 10),
               sf_dec_by_ori = c(0, 30, 60, 90, 120, 150),
               ori_dec_by_sf = c(0.02, 0.04, 0.08, 0.16, 0.32),
               adj_dist_by_sf = c(0.02, 0.04, 0.08, 0.16, 0.32, 0.64))
WT_PRE_BEFORE_TRAINING <- c("VD017", "VD018", "VD019")

prep <- function(d, metric) {
  x <- d[d$metric == metric & abs(outer(d$level, LEVELS[[metric]], "-")) %*% rep(1, length(LEVELS[[metric]])) <
           1e9 & sapply(d$level, function(v) any(abs(v - LEVELS[[metric]]) < 1e-9)), ]
  x$level <- factor(signif(x$level, 4))
  x$training <- factor(x$training, levels = c("pre", "post"))
  x$group <- factor(x$group, levels = c("WT", "Rescued"))
  x$mouse <- factor(x$mouse)
  droplevels(x)
}
art_p <- function(formula, data, term) {
  a <- anova(art(formula, data = data))
  row <- a[a$Term == term, ]
  c(F = row$F, df1 = row$Df, df2 = row$Df.res, p = row$`Pr(>F)`)
}
perm_diff <- function(a, b) {
  pooled <- c(a, b); obs <- mean(a) - mean(b)
  idx <- combn(length(pooled), length(a))
  null <- apply(idx, 2, function(i) mean(pooled[i]) - mean(pooled[-i]))
  mean(abs(null) >= abs(obs) - 1e-12)
}

run_all <- function(d, label) {
  out <- list()
  for (metric in names(LEVELS)) {
    x <- prep(d, metric)
    res <- data.frame(analysis = label, metric = metric)
    for (g in c("WT", "Rescued")) {
      xg <- droplevels(x[x$group == g, ])
      r <- art_p(value ~ training * level + (1 + training | mouse), xg, "training")
      ri <- art_p(value ~ training * level + (1 + training | mouse), xg, "training:level")
      r0 <- art_p(value ~ training * level + (1 | mouse), xg, "training")
      res[[paste0(g, "_n")]] <- nlevels(xg$mouse)
      res[[paste0(g, "_training_F")]] <- r["F"]
      res[[paste0(g, "_training_df2")]] <- r["df2"]
      res[[paste0(g, "_training_p")]] <- r["p"]
      res[[paste0(g, "_training_x_level_p")]] <- ri["p"]
      res[[paste0(g, "_training_p_(1|mouse)")]] <- r0["p"]
    }
    gx <- art_p(value ~ group * training * level + (1 + training | mouse), x, "group:training")
    res$group_x_training_F <- gx["F"]; res$group_x_training_df2 <- gx["df2"]; res$group_x_training_p <- gx["p"]
    res[["group_x_training_p_(1|mouse)"]] <- art_p(value ~ group * training * level + (1 | mouse), x,
                                                    "group:training")["p"]
    post <- droplevels(x[x$training == "post", ])
    res$post_group_p <- art_p(value ~ group * level + (1 | mouse), post, "group")["p"]
    pre <- droplevels(x[x$training == "pre", ])
    res$pre_group_p <- art_p(value ~ group * level + (1 | mouse), pre, "group")["p"]
    # mouse-level change, averaged over levels
    m <- aggregate(value ~ group + mouse + training, x, mean)
    w <- reshape(m, idvar = c("group", "mouse"), timevar = "training", direction = "wide")
    w$delta <- w$value.post - w$value.pre
    res$WT_delta_mean <- mean(w$delta[w$group == "WT"])
    res$Rescued_delta_mean <- mean(w$delta[w$group == "Rescued"])
    res$delta_perm_p <- perm_diff(w$delta[w$group == "Rescued"], w$delta[w$group == "WT"])
    out[[metric]] <- res
    if (label == "primary") {
      w$metric <- metric
      write.table(w, "mouse_deltas.csv", sep = ",", row.names = FALSE,
                  col.names = !file.exists("mouse_deltas.csv"), append = file.exists("mouse_deltas.csv"))
    }
  }
  do.call(rbind, out)
}

if (file.exists("mouse_deltas.csv")) file.remove("mouse_deltas.csv")
d <- d_all[d_all$group %in% c("WT", "Rescued"), ]
d29 <- d_all; d29$group[d29$group == "Rescued_VD029_excluded"] <- "Rescued"
dwt3 <- d[d$group == "Rescued" | d$mouse %in% WT_PRE_BEFORE_TRAINING, ]

res <- rbind(run_all(d, "primary"),
             run_all(d29, "sensitivity: VD029 included"),
             run_all(dwt3, "sensitivity: WT imaged before training only (n = 3)"))
write.csv(res, "stats_training_effect.csv", row.names = FALSE)

# per-SF pre vs post for the responsive fraction (as in Fig. 5A)
x <- prep(d, "resp_by_sf")
models <- list("(1|mouse)" = value ~ training * level + (1 | mouse),
               "(1+training|mouse)" = value ~ training * level + (1 + training | mouse))
per_sf <- do.call(rbind, lapply(names(models), function(mod) do.call(rbind, lapply(c("WT", "Rescued"), function(g) {
  xg <- droplevels(x[x$group == g, ])
  s <- as.data.frame(summary(art.con(art(models[[mod]], data = xg), "training:level", adjust = "none")))
  parts <- strsplit(as.character(s$contrast), " - ")
  keep <- sapply(parts, function(p) sub("^[^,]+,", "", p[1]) == sub("^[^,]+,", "", p[2]))
  s <- s[keep, ]
  data.frame(model = mod, group = g, contrast = s$contrast, t = s$t.ratio, p = s$p.value,
             q_BH = p.adjust(s$p.value, "BH"))
}))))
write.csv(per_sf, "stats_resp_by_sf_per_sf.csv", row.names = FALSE)

fmt <- function(p) ifelse(is.na(p), "NA", ifelse(p < 0.001, "<0.001", sprintf("%.3f", p)))
for (lab in unique(res$analysis)) {
  cat("\n=====", lab, "=====\n")
  r <- res[res$analysis == lab, ]
  print(data.frame(metric = r$metric, n = paste0(r$WT_n, "/", r$Rescued_n),
                   WT_train_p = fmt(r$WT_training_p), Resc_train_p = fmt(r$Rescued_training_p),
                   group_x_train_p = fmt(r$group_x_training_p),
                   `WT_p_(1|m)` = fmt(r[["WT_training_p_(1|mouse)"]]),
                   `Resc_p_(1|m)` = fmt(r[["Rescued_training_p_(1|mouse)"]]),
                   `gxt_p_(1|m)` = fmt(r[["group_x_training_p_(1|mouse)"]]), check.names = FALSE,
                   delta_WT = round(r$WT_delta_mean, 3),
                   delta_Resc = round(r$Rescued_delta_mean, 3), delta_perm_p = fmt(r$delta_perm_p),
                   post_group_p = fmt(r$post_group_p), pre_group_p = fmt(r$pre_group_p)), row.names = FALSE)
}
cat("\n===== responsive fraction, pre vs post per SF (ART-C, BH) =====\n")
print(transform(per_sf, t = round(t, 2), p = signif(p, 3), q_BH = signif(q_BH, 3)), row.names = FALSE)
