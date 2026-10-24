# Fig. 3M statistics: WT vs GtdKO+AAV-Gat2, early (trials 1-400) vs late (401-1000).
#
# Unit of inference is the mouse (5 WT, 5 rescued). The plotted quantity is each
# mouse's proportion correct pooled over the trials in each window, so the
# primary tests are run on exactly those values:
#   - group within epoch  : exact permutation test on the difference in means
#                           (all 252 relabellings), Holm-adjusted over 2 epochs
#   - group x epoch       : exact permutation test on per-mouse improvement
#                           (late - early); this is the test of "caught up"
#   - epoch within group  : exact sign-flip test on improvement (32 flips)
# Sensitivity analyses: ART mixed model (as used for Figs 4-6) and binomial
# GLMMs on trial-level data with and without a session random intercept.
# Everything is two-sided.

suppressPackageStartupMessages({
  library(lme4)
  library(emmeans)
  library(ARTool)
})
options(width = 130, warn = 1)

args <- commandArgs(FALSE)
f <- sub("--file=", "", grep("--file=", args, value = TRUE))
if (length(f)) setwd(dirname(normalizePath(f)))

out <- file("stats_fig3M_output.txt", open = "wt")
say <- function(...) { txt <- paste0(...); cat(txt, "\n"); writeLines(txt, out) }
show <- function(x) { txt <- capture.output(print(x)); cat(txt, sep = "\n"); writeLines(txt, out) }

tr <- read.csv("trials_clean.csv", stringsAsFactors = FALSE)
tr$keep <- as.logical(tr$keep)
d <- subset(tr, keep & epoch %in% c("early", "late") & group %in% c("WT", "Rescued"))
d$group <- factor(d$group, levels = c("WT", "Rescued"))
d$epoch <- factor(d$epoch, levels = c("early", "late"))
d$e_num <- ifelse(d$epoch == "late", 0.5, -0.5)
d$correct <- d$Correctness

# ------------------------------------------------------------------ mouse-level values
mt <- aggregate(correct ~ group + mouse + epoch, d, mean)
w <- reshape(mt, idvar = c("group", "mouse"), timevar = "epoch", direction = "wide")
names(w) <- sub("correct.", "", names(w), fixed = TRUE)
w$improvement <- w$late - w$early
w <- w[order(w$group, w$mouse), ]
write.csv(w, "mouse_level_values.csv", row.names = FALSE)
say("=== Per-mouse proportion correct (after excluding contingency-mismatch trials) ===")
show(transform(w, early = round(early, 3), late = round(late, 3), improvement = round(improvement, 3)))

is_r <- w$group == "Rescued"
splits <- combn(nrow(w), sum(is_r))                      # 252 ways to pick the "rescued" 5
perm_between <- function(v) {
  obs <- mean(v[is_r]) - mean(v[!is_r])
  null <- apply(splits, 2, function(ix) mean(v[ix]) - mean(v[-ix]))
  c(diff = obs, p = mean(abs(null) >= abs(obs) - 1e-12))
}
flips <- as.matrix(expand.grid(rep(list(c(-1, 1)), sum(is_r))))
perm_within <- function(v) {
  obs <- mean(v)
  null <- apply(flips, 1, function(s) mean(s * v))
  c(diff = obs, p = mean(abs(null) >= abs(obs) - 1e-12))
}
welch_ci <- function(a, b, level) t.test(a, b, conf.level = level)$conf.int * 100
pw_e <- perm_between(w$early); pw_l <- perm_between(w$late); pw_i <- perm_between(w$improvement)
pr <- perm_within(w$improvement[is_r]); pwt <- perm_within(w$improvement[!is_r])

prim <- data.frame(
  test = c("Rescued vs WT, trials 1-400", "Rescued vs WT, trials 401-1000",
           "Group x epoch: improvement, Rescued vs WT",
           "Improvement within Rescued", "Improvement within WT"),
  diff_pp = 100 * c(pw_e["diff"], pw_l["diff"], pw_i["diff"], pr["diff"], pwt["diff"]),
  ci95 = c(sprintf("[%.1f, %.1f]", welch_ci(w$early[is_r], w$early[!is_r], .95)[1], welch_ci(w$early[is_r], w$early[!is_r], .95)[2]),
           sprintf("[%.1f, %.1f]", welch_ci(w$late[is_r], w$late[!is_r], .95)[1], welch_ci(w$late[is_r], w$late[!is_r], .95)[2]),
           sprintf("[%.1f, %.1f]", welch_ci(w$improvement[is_r], w$improvement[!is_r], .95)[1], welch_ci(w$improvement[is_r], w$improvement[!is_r], .95)[2]),
           sprintf("[%.1f, %.1f]", 100 * t.test(w$improvement[is_r])$conf.int[1], 100 * t.test(w$improvement[is_r])$conf.int[2]),
           sprintf("[%.1f, %.1f]", 100 * t.test(w$improvement[!is_r])$conf.int[1], 100 * t.test(w$improvement[!is_r])$conf.int[2])),
  exact_p = c(pw_e["p"], pw_l["p"], pw_i["p"], pr["p"], pwt["p"]),
  n_perm = c(252, 252, 252, 32, 32))
prim$p_holm <- c(p.adjust(prim$exact_p[1:2], "holm"), NA, p.adjust(prim$exact_p[4:5], "holm"))
prim$min_attainable_p <- c(2 / 252, 2 / 252, 2 / 252, 2 / 32, 2 / 32)
say("\n=== PRIMARY: exact mouse-level tests (two-sided; 95% CI = Welch/t, descriptive) ===")
show(format(prim, digits = 3))

ci90_late <- welch_ci(w$late[is_r], w$late[!is_r], .90)
say(sprintf("Equivalence (trials 401-1000): 90%% CI [%.1f, %.1f] pp vs +/-10 pp margin -> %s",
            ci90_late[1], ci90_late[2],
            ifelse(ci90_late[1] > -10 & ci90_late[2] < 10, "equivalent", "equivalence NOT established")))

wil <- sapply(c("early", "late", "improvement"), function(v)
  wilcox.test(w[[v]][is_r], w[[v]][!is_r], exact = TRUE)$p.value)
say(sprintf("Exact Wilcoxon rank-sum (two-sided): early p = %.4f, late p = %.4f, improvement p = %.4f",
            wil[1], wil[2], wil[3]))

# ------------------------------------------------------------------ ART (as in Figs 4-6)
wl <- rbind(data.frame(group = w$group, mouse = w$mouse, epoch = "early", value = w$early),
            data.frame(group = w$group, mouse = w$mouse, epoch = "late", value = w$late))
wl$group <- factor(wl$group, levels = c("WT", "Rescued")); wl$epoch <- factor(wl$epoch)
wl$mouse <- factor(wl$mouse)
art_m <- art(value ~ group * epoch + (1 | mouse), data = wl)
say("\n=== SENSITIVITY 1: ART mixed model, value ~ group * epoch + (1 | mouse) ===")
show(anova(art_m))
art_all <- as.data.frame(summary(art.con(art_m, "group:epoch", adjust = "none")))
art_g <- art_all[art_all$contrast %in% c("Rescued,early - WT,early", "Rescued,late - WT,late"), ]
art_e <- art_all[art_all$contrast %in% c("Rescued,early - Rescued,late", "WT,early - WT,late"), ]
art_g$p_holm <- p.adjust(art_g$p.value, "holm")
art_e$p_holm <- p.adjust(art_e$p.value, "holm")
say("ART-C contrasts (aligned-rank units; Holm within each family):")
show(format(rbind(art_g, art_e), digits = 3))

# ------------------------------------------------------------------ binomial GLMMs
ctrl <- glmerControl(optimizer = "bobyqa", optCtrl = list(maxfun = 2e5))
glmm_block <- function(formula, label) {
  m1 <- glmer(formula, data = d, family = binomial, control = ctrl)
  m0 <- update(m1, . ~ . - group:epoch)
  p_int <- anova(m0, m1)$`Pr(>Chisq)`[2]
  emm <- emmeans(m1, ~ group * epoch)
  g <- summary(contrast(emm, method = list("Rescued/WT" = c(-1, 1)), by = "epoch"),
               infer = TRUE, type = "response")
  e <- summary(contrast(emm, method = list("late/early" = c(-1, 1)), by = "group"),
               infer = TRUE, type = "response")
  est <- summary(emm, type = "response")
  say("\n=== ", label, " ===")
  say("formula: ", deparse(formula), "   singular: ", isSingular(m1))
  vc <- as.data.frame(VarCorr(m1))
  say("random-effect SDs (logit): ", paste(sprintf("%s:%s=%.2f", vc$grp,
      ifelse(is.na(vc$var2), vc$var1, "corr"), vc$sdcor), collapse = "  "))
  say("model-estimated P(correct): ", paste(sprintf("%s-%s %.3f", est$group, est$epoch, est$prob), collapse = ", "))
  say(sprintf("group x epoch LRT p = %.4f", p_int))
  say(sprintf("Rescued vs WT, %s: OR %.2f [%.2f, %.2f], p = %.4f (Holm %.4f)", g$epoch, g$odds.ratio,
              g$asymp.LCL, g$asymp.UCL, g$p.value, p.adjust(g$p.value, "holm")))
  say(sprintf("late vs early, %s: OR %.2f [%.2f, %.2f], p = %.4f (Holm %.4f)", e$group, e$odds.ratio,
              e$asymp.LCL, e$asymp.UCL, e$p.value, p.adjust(e$p.value, "holm")))
  invisible(list(p_int = p_int, g = g, e = e))
}
g1 <- glmm_block(correct ~ group * epoch + (1 + e_num | mouse),
                 "SENSITIVITY 2: binomial GLMM, mouse random intercept + epoch slope")
g2 <- glmm_block(correct ~ group * epoch + (1 + e_num | mouse) + (1 | session),
                 "SENSITIVITY 3: binomial GLMM, + session random intercept (weights sessions ~equally)")

# ------------------------------------------------------------------ values without trial exclusions
o <- read.csv("mouse_level_values_ORIGINAL.csv", stringsAsFactors = FALSE)
say("\n=== Analysis without trial exclusions, for reference ===")
for (v in c("early", "late")) {
  a <- o[[v]][o$group == "WT"]; b <- o[[v]][o$group == "Rescued"]
  say(sprintf("  %-5s WT %.3f vs Rescued %.3f | rank-sum two-sided p = %.4f | one-sided (WT > Rescued) p = %.4f",
              v, mean(a), mean(b), wilcox.test(a, b, exact = TRUE)$p.value,
              wilcox.test(a, b, exact = TRUE, alternative = "greater")$p.value))
}

write.csv(prim, "stats_primary_exact.csv", row.names = FALSE)
close(out)
