# Lancet house style for ggplot2, rendered through proper devices.
#
# Runs on IU HPC Quartz (module r/4.6.1) with svglite, ragg, patchwork and
# systemfonts from a user library. systemfonts resolves "Arial" to Liberation
# Sans, which is metrically identical to Arial, so the type sets at the widths a
# Lancet figure expects.
#
# The forests are drawn in the form this journal prints, which is a table whose
# middle column happens to be a picture: a label column on the left, bold
# headings with plain rows under them, the plot, then the estimate in figures.
# It is set in black on white. Colour is spent only where position cannot carry
# the distinction, which in this manuscript is the per-cohort risk curves.

# The cluster starts R in the C locale, which drops a literal en dash and a
# literal middle dot from a format string. Set a UTF-8 locale, and ask for both
# by codepoint.
suppressWarnings(Sys.setlocale("LC_ALL", "C.UTF-8"))
EN_DASH <- "\u2013"
SUP2    <- "\u00b2"   # the same trap: a literal superscript two is dropped too
MIDDOT  <- "\u00b7"

suppressPackageStartupMessages({
  library(ggplot2); library(patchwork); library(grid)
})

OKABE <- c(black = "#000000", orange = "#E69F00", sky = "#56B4E9",
           green = "#009E73", yellow = "#F0E442", blue = "#0072B2",
           vermilion = "#D55E00", purple = "#CC79A7", grey = "#999999")

INK   <- "#000000"
RULE  <- "#000000"

# Mix a Wong hue toward a light neutral, so the marker carries a calm colour
# rather than a saturated one. Derived rather than typed, so the relationship to
# the palette stays visible.
lighten <- function(hex, toward = "#E8E8E8", by = 0.5) {
  a <- grDevices::col2rgb(hex); b <- grDevices::col2rgb(toward)
  grDevices::rgb(t(round(a * (1 - by) + b * by)), maxColorValue = 255)
}

# The marker this journal prints: a light filled square with the interval
# passing through it and a tick at the point estimate. The square is large
# enough to read at 183 mm, and its colour is a muted Wong sky blue; magnitude
# is still carried by position alone, so the panel survives greyscale.
MARK_FILL <- NULL   # set below, once OKABE exists
MARK_EDGE <- NULL
MARK_SIZE <- 2.50
REFERENCE  <- unname(OKABE["grey"])
POOLED     <- unname(OKABE["black"])

MARK_FILL <- lighten(unname(OKABE["sky"]))
MARK_EDGE <- unname(OKABE["blue"])

COHORT_COL <- c(CHARLS = unname(OKABE["vermilion"]), ELSA = unname(OKABE["blue"]),
                HRS = unname(OKABE["orange"]), KLoSA = unname(OKABE["green"]),
                SHARE = unname(OKABE["purple"]))
COHORT_PCH <- c(CHARLS = 25, ELSA = 21, HRS = 22, KLoSA = 24, SHARE = 23)

# Apparent size, which is not the same as the size set here. These drawings are
# 183 mm wide, and the manuscript page they are embedded in has 6.50 inches of
# text width, so Word draws them at about 90% and every label loses a tenth of
# its height on the way. They were embedded at 5.83 inches for a while, which is
# 81%, and put the row labels on the page at 5.7 pt: small enough that a
# clinician reading the submitted file said so. The type is set here for what
# reaches the page after that reduction, and the row pitch moves with it so a
# larger label cannot crowd the row above.

SINGLE_COL <- 3.60   # inches; two of these tile to the 183 mm double column
DOUBLE_COL <- 7.20
BASE_SIZE  <- 9      # see the note on apparent size below
SMALL_SIZE <- 8      # table rows, tick labels, column headers
FONT       <- "Arial"

# Lancet sets decimals on the middle dot, in the figures as well as the text.
md <- function(s) gsub(".", MIDDOT, s, fixed = TRUE)
fmt <- function(x, digits = 2) md(formatC(x, format = "f", digits = digits))
fmt_break <- function(x) md(formatC(x, format = "fg"))
ci_text <- function(est, lo, hi, digits = 2, ref = "(ref)")
  ifelse(is.na(lo), paste(fmt(est, digits), ref),
         paste0(fmt(est, digits), " (", fmt(lo, digits), EN_DASH,
                fmt(hi, digits), ")"))

# Applied after theme_lancet(), which would otherwise restore the axis line.
no_y_axis <- function() theme(axis.line.y = element_blank(),
                              axis.ticks.y = element_blank())

theme_lancet <- function(base_size = BASE_SIZE) {
  theme_classic(base_size = base_size, base_family = FONT) +
    theme(
      axis.line        = element_line(linewidth = 0.25, colour = "black"),
      axis.ticks       = element_line(linewidth = 0.25, colour = "black"),
      axis.ticks.length = unit(1.8, "pt"),
      axis.text        = element_text(size = SMALL_SIZE, colour = "black"),
      axis.title       = element_text(size = base_size, colour = "black"),
      legend.position  = "bottom",
      legend.title     = element_blank(),
      legend.text      = element_text(size = SMALL_SIZE),
      legend.key.size  = unit(10, "pt"),
      legend.margin    = margin(t = -2),
      legend.background = element_blank(),
      legend.key       = element_blank(),
      plot.tag         = element_text(size = base_size + 1, face = "bold",
                                      family = FONT, hjust = 0, vjust = 1),
      plot.tag.position = c(0.005, 0.995),
      plot.margin      = margin(4, 4, 2, 2),
      plot.background  = element_rect(fill = "white", colour = NA),
      panel.background = element_rect(fill = "white", colour = NA),
      strip.background = element_blank()
    )
}

# --------------------------------------------------------------- the forest
#
# rows: one line per printed line, top to bottom.
#   label     the text in the left column
#   heading   TRUE for a bold group heading, FALSE for an estimate row
#   estimate, lo, hi   NA on a heading row; lo and hi NA on a reference row
#   extra     optional right-hand column, character
#
# Everything sits on one linear x axis running 0 to 1 and the ratio scale is
# mapped into the plot column by hand, because the columns either side of it are
# text rather than data and one ggplot scale cannot serve both. The reward is
# that the column edges are declared numbers, so the layout is inspectable
# rather than emergent.
# One line of SMALL_SIZE type, expressed in row units at the pitch below. The
# furniture under the axis is placed in these units, and the figure is then
# sized from the unit count, so a tick label can never land on an axis title
# because the panel happened to be short.
ROW_PITCH <- 0.177   # inches per row
LINE_U    <- 0.63    # row units occupied by one line of SMALL_SIZE type
TOP_U     <- 0.65    # above the column headers

# How tall a forest wants to be, in row units and then in inches. The caller
# uses this to size the figure and to set patchwork heights, so every forest in
# the manuscript keeps the same row pitch whatever its row count.
forest_units <- function(n, xlab = NULL, favours = NULL) {
  foot <- if (!is.null(favours)) 1.55 + LINE_U + 0.25
          else if (!is.null(xlab))
            1.15 + LINE_U * length(strsplit(xlab, "\n")[[1]]) + 0.25
          else 0.32 + LINE_U + 0.25
  TOP_U + n + 0.95 + foot
}
forest_height <- function(n, xlab = NULL, favours = NULL)
  ROW_PITCH * forest_units(n, xlab, favours) + 0.10

lancet_forest <- function(rows, breaks, limits, est_header, xlab = NULL,
                          plot_x = c(0.36, 0.66), est_x = 0.90,
                          extra_x = NULL, extra_header = NULL,
                          favours = NULL, indent = 0.018) {
  stopifnot(is.data.frame(rows), all(c("label", "heading") %in% names(rows)))
  est <- rows[!rows$heading, ]
  if (any(!is.na(est$lo) & est$lo < limits[1]) ||
      any(!is.na(est$hi) & est$hi > limits[2]))
    stop("an interval falls outside the axis; widen limits rather than clipping")

  to_x <- function(v) plot_x[1] + diff(plot_x) *
    (log(v) - log(limits[1])) / (log(limits[2]) - log(limits[1]))

  n <- nrow(rows)
  rows$y <- -seq_len(n)
  est <- rows[!rows$heading, ]
  head_y   <- 0.15
  rule_y   <- -0.30
  axis_y   <- -(n + 0.95)
  tick_y   <- axis_y - 0.15
  ticklab_y <- axis_y - 0.32
  xlab_y   <- axis_y - 1.15
  fav_y    <- axis_y - 1.35
  favtxt_y <- axis_y - 1.55
  top_y    <- TOP_U
  bot_y    <- top_y - forest_units(n, xlab, favours)
  sz <- SMALL_SIZE / .pt

  p <- ggplot() +
    annotate("segment", x = 0, xend = 1, y = rule_y, yend = rule_y,
             colour = RULE, linewidth = 0.3) +
    annotate("segment", x = to_x(1), xend = to_x(1), y = rule_y, yend = axis_y,
             colour = REFERENCE, linetype = "22", linewidth = 0.25) +
    annotate("text", x = est_x, y = head_y, label = est_header, hjust = 1,
             size = sz, family = FONT, colour = INK) +
    # square first, then the interval through it, then the tick at the estimate:
    # the order is what produces the cross inside the marker that this journal
    # prints, and it keeps the estimate readable inside a light square
    geom_point(data = est, aes(x = to_x(estimate), y = y),
               shape = 22, fill = MARK_FILL, colour = MARK_EDGE,
               size = MARK_SIZE, stroke = 0.32) +
    geom_segment(data = est[!is.na(est$lo), ],
                 aes(x = to_x(lo), xend = to_x(hi), y = y, yend = y),
                 colour = INK, linewidth = 0.34) +
    geom_segment(data = est, aes(x = to_x(estimate), xend = to_x(estimate),
                                 y = y - 0.19, yend = y + 0.19),
                 colour = INK, linewidth = 0.34) +
    geom_text(data = rows[rows$heading, ], aes(x = 0, y = y, label = label),
              hjust = 0, size = sz, family = FONT, fontface = "bold",
              colour = INK) +
    geom_text(data = est, aes(x = indent, y = y, label = label), hjust = 0,
              size = sz, family = FONT, colour = INK) +
    geom_text(data = est, aes(x = est_x, y = y, label = text), hjust = 1,
              size = sz, family = FONT, colour = INK) +
    annotate("segment", x = to_x(limits[1]), xend = to_x(limits[2]),
             y = axis_y, yend = axis_y, colour = RULE, linewidth = 0.3) +
    annotate("segment", x = to_x(breaks), xend = to_x(breaks),
             y = axis_y, yend = tick_y, colour = RULE, linewidth = 0.3) +
    annotate("text", x = to_x(breaks), y = ticklab_y, label = fmt_break(breaks),
             size = sz, family = FONT, colour = INK, vjust = 1) +
    coord_cartesian(xlim = c(0, 1), ylim = c(bot_y, top_y),
                    expand = FALSE, clip = "off") +
    theme_void(base_family = FONT) +
    theme(plot.margin = margin(3, 3, 2, 2),
          plot.tag = element_text(size = BASE_SIZE + 1, face = "bold",
                                  family = FONT, hjust = 0, vjust = 1),
          plot.background = element_rect(fill = "white", colour = NA))

  if (!is.null(extra_x)) {
    p <- p +
      annotate("text", x = extra_x, y = head_y, label = extra_header, hjust = 1,
               size = sz, family = FONT, colour = INK) +
      geom_text(data = est, aes(x = extra_x, y = y, label = extra), hjust = 1,
                size = sz, family = FONT, colour = INK)
  }

  if (!is.null(favours)) {
    # the direction cue this journal prints under a forest axis. No axis title
    # goes with it: the estimate column already names the quantity.
    arw <- arrow(length = unit(2.6, "pt"), ends = "both", type = "closed")
    p <- p +
      annotate("segment", x = to_x(limits[1]), xend = to_x(limits[2]),
               y = fav_y, yend = fav_y, colour = INK, linewidth = 0.3,
               arrow = arw) +
      annotate("text", x = to_x(limits[1]), y = favtxt_y, label = favours[1],
               hjust = 0, size = sz, family = FONT, colour = INK, vjust = 1) +
      annotate("text", x = to_x(limits[2]), y = favtxt_y, label = favours[2],
               hjust = 1, size = sz, family = FONT, colour = INK, vjust = 1)
  } else if (!is.null(xlab)) {
    p <- p + annotate("text", x = mean(plot_x), y = xlab_y, label = xlab,
                      size = sz, family = FONT, colour = INK, vjust = 1,
                      lineheight = 0.95)
  }
  p
}

# Rows for a forest, assembled group by group, so the caller writes the table it
# wants to see rather than arithmetic on row indices.
forest_rows <- function(...) {
  parts <- list(...)
  do.call(rbind, lapply(parts, function(g) {
    body <- g$rows
    body$heading <- FALSE
    head <- body[1, ]
    head[] <- NA
    head$label <- g$heading
    head$heading <- TRUE
    if (is.na(g$heading)) body else rbind(head, body)
  }))
}

group_block <- function(heading, label, estimate, lo, hi, extra = NA_character_,
                        digits = 2) {
  list(heading = heading,
       rows = data.frame(label = label, estimate = estimate, lo = lo, hi = hi,
                         text = ci_text(estimate, lo, hi, digits),
                         extra = as.character(extra),
                         stringsAsFactors = FALSE))
}

# The forest x axis for the panels that are still ordinary plots.
scale_rr <- function(breaks, limits) {
  scale_x_log10(breaks = breaks, labels = fmt_break(breaks),
                limits = limits, expand = expansion(0))
}
null_line <- function() geom_vline(xintercept = 1, colour = REFERENCE,
                                   linetype = "22", linewidth = 0.25)

# One drawing, three devices. svglite writes real <text> elements rather than
# glyph outlines, so PowerPoint can convert the result to editable shapes;
# ragg renders the raster; cairo_pdf embeds the font in the vector file.
save_lancet <- function(plot, stem, width, height, dpi = 600) {
  dir.create(dirname(stem), showWarnings = FALSE, recursive = TRUE)
  # The manifest is what the checker measures against. The declared sizes are
  # computed from the row counts now, so keeping a second copy of them in the
  # checker would be a copy that goes stale the first time a forest gains a row.
  manifest <- file.path(dirname(stem), "sizes.csv")
  if (!file.exists(manifest))
    cat("stem,width,height\n", file = manifest)
  cat(sprintf("%s,%.4f,%.4f\n", basename(stem), width, height),
      file = manifest, append = TRUE)
  svglite::svglite(paste0(stem, ".svg"), width = width, height = height,
                   pointsize = BASE_SIZE, bg = "white")
  print(plot); dev.off()
  ragg::agg_png(paste0(stem, ".png"), width = width, height = height,
                units = "in", res = dpi, background = "white")
  print(plot); dev.off()
  cairo_pdf(paste0(stem, ".pdf"), width = width, height = height,
            pointsize = BASE_SIZE, bg = "white")
  print(plot); dev.off()
  invisible(stem)
}

read_panel <- function(name) read.csv(file.path(DATA, name), stringsAsFactors = FALSE)
