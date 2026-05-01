# ==============================================================================
# Governance of Asia Web Interface
# ==============================================================================
# Author:       [Anonymized for review]
# Institute:    [Anonymized for review]
# Year:         2026
# License:      CC BY-SA 4.0
#
# R Version:    4.5.2+
# ==============================================================================

# ------------------------------------------------------------------------------
# Required Libraries
# ------------------------------------------------------------------------------

# Core Shiny packages
library(shiny)            # Web application framework
library(shinyjs)          # JavaScript operations in Shiny
library(shinythemes)      # Bootstrap themes for Shiny
library(shinyWidgets)     # Custom input widgets
library(shinycssloaders)  # Loading animations
library(htmlwidgets)      # HTML widget framework
library(bslib)            # Bootstrap styling

# Data manipulation and analysis
library(tidyverse)        # Data wrangling ecosystem
library(jsonlite)
library(DT)
library(tmcn)

# Network and graph analysis
library(igraph)           # Network analysis
library(visNetwork)       # Interactive network visualization

# Visualization packages
library(ggplot2)          # Grammar of graphics
library(ggtext)           # Enhanced text rendering for ggplot2
library(ggforce)
library(ggrepel)          # Label positioning for ggplot2
library(ggridges)
library(fmsb)
library(treemapify)       # Treemap geom for ggplot2
library(viridisLite)      # Color scales for visualization

# Geospatial analysis
library(sf)               # Simple features for spatial data

# File export
library(writexl)          # Excel file export

# ------------------------------------------------------------------------------
# Application Configuration
# ------------------------------------------------------------------------------

# Set maximum file size
options(shiny.maxRequestSize = 100 * 1024^2)
options(bslib.precompiled = TRUE)
options(sass.cache = TRUE)
options(shiny.minified = TRUE)

# ------------------------------------------------------------------------------
# Data Mappings and Configuration
# ------------------------------------------------------------------------------

# known_fields <- c("civil", "mechanical", "electrical", "mining", "chemical", "textile")

# ------------------------------------------------------------------------------
# Data Loading — from pre-binarized RDS (run binarize.R first)
# ------------------------------------------------------------------------------

if (!file.exists("app_data.rds")) {
  stop("app_data.rds not found. Run 'Rscript binarize.R' first to generate it.")
}

app_data <- readRDS("app_data.rds")

# Unpack all objects into the global environment
map_china              <- app_data$map_china
map_taiwan             <- app_data$map_taiwan
map_japan              <- app_data$map_japan
map_korea              <- app_data$map_korea
map_asia_outer         <- app_data$map_asia_outer
map_asia               <- app_data$map_asia
province_country       <- app_data$province_country
manchukuo_provinces    <- app_data$manchukuo_provinces
religions_dict         <- app_data$religions_dict
religion_category_map  <- app_data$religion_category_map
hobbies_dict           <- app_data$hobbies_dict
relations_dict         <- app_data$relations_dict
records_locations      <- app_data$records_locations
records_persons_core   <- app_data$records_persons_core
records_persons        <- app_data$records_persons
records_ranks          <- app_data$records_ranks
records_hobbies        <- app_data$records_hobbies
records_religion       <- app_data$records_religion
records_political_parties <- app_data$records_political_parties
records_family         <- app_data$records_family
records_family_education  <- app_data$records_family_education
records_family_career  <- app_data$records_family_career
records_organizations  <- app_data$records_organizations
records_education      <- app_data$records_education
records_career         <- app_data$records_career
person_career_countries <- app_data$person_career_countries
records_appearances    <- app_data$records_appearances

if (is.null(records_appearances)) {
  records_appearances <- jsonlite::stream_in(file("data/person_appearances.jsonl"), simplifyVector = TRUE)
  records_appearances$page <- suppressWarnings(as.integer(sub("^[^_]*_0*", "", records_appearances$source_page)))
  records_appearances <- records_appearances[, c("person_id", "volume", "source_page", "page")]
  records_appearances <- unique(records_appearances)
}

rm(app_data)  # free memory

# NDL pid + display label + actual publication year per Taishū jinji-roku volume
ndl_volume_meta <- list(
  "1927"  = list(pid = "1688498", label = "1927",            year = 1927L),
  "1935"  = list(pid = "8312058", label = "1935",            year = 1935L),
  "1943A" = list(pid = "1229896", label = "1940 (vol. 14a)", year = 1940L),
  "1943B" = list(pid = "1683373", label = "1942 (vol. 14b)", year = 1942L),
  "1943D" = list(pid = "1229971", label = "1943 (vol. 14d)", year = 1943L),
  "1943E" = list(pid = "1230025", label = "1943 (vol. 14e)", year = 1943L)
)
volume_sort_order <- c("1927" = 1, "1935" = 2, "1943A" = 3, "1943B" = 4, "1943D" = 5, "1943E" = 6)
volume_year_lookup <- setNames(
  vapply(ndl_volume_meta, `[[`, integer(1), "year"),
  names(ndl_volume_meta)
)

format_person_sources <- function(pid) {
  rows <- records_appearances[records_appearances$person_id == pid, , drop = FALSE]
  if (nrow(rows) == 0) return(NULL)
  rows <- rows[order(volume_sort_order[rows$volume], rows$page), ]
  items <- vapply(seq_len(nrow(rows)), function(i) {
    meta <- ndl_volume_meta[[rows$volume[i]]]
    if (is.null(meta) || is.na(rows$page[i])) return(NA_character_)
    paste0('<a href="https://dl.ndl.go.jp/pid/', meta$pid, '/1/', rows$page[i],
           '" target="_blank">', meta$label, ', p. ', rows$page[i], '</a>')
  }, character(1))
  items <- items[!is.na(items)]
  if (length(items) == 0) return(NULL)
  HTML(paste0(
    "<p style='text-align:center; margin-top:0.5em; font-size:0.9em;'>",
    "<b>Compare & complete using original NDL scans:</b> ",
    paste(items, collapse = " · "),
    "</p>"
  ))
}

# Helper: format location as "English City, English District 日本語"
format_location <- function(loc_row) {
  if (is.null(loc_row) || nrow(loc_row) == 0) return("")
  en_parts <- c()
  if (!is.na(loc_row$admin1_en[1]) && nzchar(loc_row$admin1_en[1]))
    en_parts <- c(en_parts, loc_row$admin1_en[1])
  if (!is.na(loc_row$admin2_en[1]) && nzchar(loc_row$admin2_en[1]))
    en_parts <- c(en_parts, loc_row$admin2_en[1])
  en_str <- paste(en_parts, collapse = ", ")
  ja_str <- loc_row$location_name[1]
  if (nzchar(en_str) && !is.na(ja_str) && nzchar(ja_str)) paste(en_str, ja_str)
  else if (!is.na(ja_str) && nzchar(ja_str)) ja_str
  else en_str
}

# ==============================================================================
# UI DEFINITION
# ==============================================================================

ui <- fluidPage(
  useShinyjs(),
  tags$head(
    tags$script(HTML("
  $(document).on('click', '.college-link', function(e) {
    e.preventDefault();
    const cid = $(this).data('id');
    Shiny.setInputValue('clicked_college_id', cid, {priority: 'event'});
  });

  $(document).on('click', '.employer-link', function(e) {
    e.preventDefault();
    const eid = $(this).data('id');
    Shiny.setInputValue('clicked_employer_id', eid, {priority: 'event'});
  });
  
    $(document).on('click', '.subsidy-link', function(e) {
    e.preventDefault();
    const eid = $(this).data('id');
    Shiny.setInputValue('clicked_subsidy_id', eid, {priority: 'event'});
  });
  
      $(document).on('click', '.society-link', function(e) {
    e.preventDefault();
    const eid = $(this).data('id');
    Shiny.setInputValue('clicked_society', eid, {priority: 'event'});
  });
  
      $(document).on('click', '.entry-link', function(e) {
    e.preventDefault();
    const eid = $(this).data('id');
    Shiny.setInputValue('clicked_entry_id', eid, {priority: 'event'});
  });

      $(document).on('click', '.location-link', function(e) {
    e.preventDefault();
    const lid = $(this).data('id');
    Shiny.setInputValue('clicked_location_id', lid, {priority: 'event'});
  });

  $(document).on('click', '.nav-person-link', function(e) {
    e.preventDefault();
    const pid = $(this).data('id');
    Shiny.setInputValue('nav_person_id', pid, {priority: 'event'});
  });

  $(document).on('click', '.nav-employer-link', function(e) {
    e.preventDefault();
    const eid = $(this).data('id');
    Shiny.setInputValue('nav_employer_id', eid, {priority: 'event'});
  });

  $(document).on('click', '.nav-college-link', function(e) {
    e.preventDefault();
    const cid = $(this).data('id');
    Shiny.setInputValue('nav_college_id', cid, {priority: 'event'});
  });

  $(document).on('click', '.network-focus-link', function(e) {
    e.preventDefault();
    const nid = $(this).data('id');
    Shiny.setInputValue('focus_network_node', nid, {priority: 'event'});
  });

  $(document).on('click', '#reset_button, #go', function() {
    $('#filter_loading').show();
  });

  // --- Experimental-website disclaimer with cookie-based opt-out ---
  function goaGetCookie(name) {
    var match = document.cookie.match(new RegExp('(?:^|; )' + name + '=([^;]*)'));
    return match ? decodeURIComponent(match[1]) : null;
  }
  function goaSetCookie(name, value, days) {
    var d = new Date();
    d.setTime(d.getTime() + days * 24 * 60 * 60 * 1000);
    document.cookie = name + '=' + encodeURIComponent(value) +
      '; expires=' + d.toUTCString() + '; path=/; SameSite=Lax';
  }
  function goaShowDisclaimer() {
    if (goaGetCookie('goa_disclaimer_dismissed') === '1') return;
    if (document.getElementById('goa_disclaimer_overlay')) return;
    var overlay = document.createElement('div');
    overlay.id = 'goa_disclaimer_overlay';
    overlay.style.cssText = 'position:fixed;inset:0;background:rgba(0,0,0,0.55);' +
      'z-index:10000;display:flex;align-items:center;justify-content:center;';
    overlay.innerHTML =
      '<div style=\"background:#fff;max-width:520px;width:90%;padding:24px 28px;' +
      'border-radius:6px;box-shadow:0 4px 24px rgba(0,0,0,0.25);font-family:inherit;\">' +
        '<h4 style=\"margin-top:0;color:#a94442;\">Experimental research prototype</h4>' +
        '<p style=\"font-size:0.95em;line-height:1.45;\">The dataset presented through this dashboard was created using a ' +
        'small language model (AI) and may contain mistakes, errors, or omissions. English readings ' +
        'and translations of names, places, and organisations were machine-generated and have ' +
        'not been fully reviewed. Please verify any finding against the original sources before ' +
        'citing or relying on it.</p>' +
        '<label style=\"display:flex;align-items:center;gap:6px;font-size:0.9em;margin:14px 0 18px 0;cursor:pointer;\">' +
          '<input type=\"checkbox\" id=\"goa_disclaimer_dontshow\"> Do not display again (requires cookie)' +
        '</label>' +
        '<div style=\"text-align:right;\">' +
          '<button id=\"goa_disclaimer_ok\" type=\"button\" class=\"btn btn-primary\">I understand</button>' +
        '</div>' +
      '</div>';
    document.body.appendChild(overlay);
    document.getElementById('goa_disclaimer_ok').addEventListener('click', function() {
      if (document.getElementById('goa_disclaimer_dontshow').checked) {
        goaSetCookie('goa_disclaimer_dismissed', '1', 365);
      }
      overlay.remove();
    });
  }
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', goaShowDisclaimer);
  } else {
    goaShowDisclaimer();
  }

  ")),
    
    tags$meta(charset = "UTF-8"),
    tags$meta(name = "author", content = "[Anonymized for review]"),
    tags$meta(name = "description", content = "Governance of Asia is a historical biographical database of actors in Japanese-controlled Asia."),
    tags$meta(property = "og:title", content = "Governance of Asia"),
    tags$meta(property = "og:description", content = "Governance of Asia is a historical biographical database of actors in Japanese-controlled Asia."),
    tags$meta(property = "og:type", content = "website"),
    
    # Favicon for standard browsers
    tags$link(rel = "icon", type = "image/x-icon", href = "favicon.ico"),
    # Mobile Web App Icons
    tags$link(rel = "apple-touch-icon", sizes = "180x180", href = "apple-touch-icon.png"),
    tags$link(rel = "icon", type = "image/png", sizes = "32x32", href = "favicon-32x32.png"),
    tags$link(rel = "icon", type = "image/png", sizes = "16x16", href = "favicon-16x16.png"),
    
    tags$script(HTML("
    $(document).on('shown.bs.collapse', function (e) {
      $(e.target).prev('.panel-heading').find('i.fa').removeClass('fa-plus').addClass('fa-minus');
    });
    $(document).on('hidden.bs.collapse', function (e) {
      $(e.target).prev('.panel-heading').find('i.fa').removeClass('fa-minus').addClass('fa-plus');
    });
  ")),
    
    # Custom CSS styling
    tags$style(HTML("
    
/* --- Force full-viewport width (robust fallback) --- */
 html, body {
   box-shadow: none !important;
   font-size: 14px;
     margin: 0;
     padding: 0;
     height: 100%;
     min-height: 100vh;
     width: 100%;
     box-sizing: border-box;
 }
/* Make sure common high-level containers won't limit width */
 #page-wrapper, .container, .container-fluid, .app-wrapper, .shiny-app {
     display: block !important;
    /* avoid inline-block shrinkage */
     width: 100% !important;
    /* always take full width */
     min-width: 100% !important;
     max-width: 100% !important;
    /* override any Bootstrap max-width */
     box-sizing: border-box;
}
/* If body is ever set to flex (your media queries do this), ensure the child grows */
 body {
    /* If you need body as flex for wide-aspect centering, keep it. If not, this keeps body a normal block. */
    /* display: flex;
     justify-content:center;
     */
 }
 body > #page-wrapper, body > .container, body > .container-fluid {
     flex: 1 1 auto;
    /* allow wrapper to expand inside a flex body */
     align-self: stretch;
 }
 
 h4 {
        font-size: 1.4rem;
        font-weight: 550;
 }
 
 .footer {
 font-family: 'Open Sans', -apple-system, BlinkMacSystemFont, Segoe UI, Roboto, Helvetica Neue, Arial, sans-serif !default;
        position: fixed;
        headings-font-weight: 300 !default;
        left: 0;
        bottom: 0;
        width: 100%;
        background-color: #ff5964;
        color: #fff;
        text-align: center;
        padding: 10px;
        border-top: 1px solid #ccc;
        a,
a:link,
a:visited,
a:hover,
a:active {
  color: #fff !important;
  text-decoration: none; /* optional: removes underline */
}
 }
    
 @media(min-aspect-ratio: 16/9) {
     body {
         display: flex;
         justify-content: center;
    }
     #page-wrapper {
         max-width: calc(100vh * (16 / 9));
         width: 100%;
    }
 }
/*  .irs-from, .irs-to {
  display: none !important;
} */
/* Dynamic Heights */
 .dynamic-height {
     height: 90vh;
     margin: 0 auto;
    /* center horizontally */
}
 .dynamic-height-network {
     height: 85vh !important;
}
 .full-width-plot {
     width: 100% !important;
     padding: 0 !important;
     margin: 0 !important;
}
/* Scrollable Content */
 .scrollable-content {
     flex-grow: 1;
     overflow-y: auto;
     padding: 0;
     margin: 0;
     overflow-x: auto;
    /* allow horizontal scroll */
     -webkit-overflow-scrolling: touch;
    /* smooth on mobile */
}
/* Centered Content */
 .center-content {
     width: 100%;
     justify-content: top;
     justify-content: flex-start;
     align-items: center;
     margin: 0;
     padding: 0;
 }
 .center-content button {
  display: block;
  margin: 0;
 }
.well {
  box-sizing: border-box;
}
    
.center-content p {
  margin: .1em;
  padding: .1em;
}
/* Navbar Styling */
 .navbar {
     background-color: #ff5964;
}
/* Hover Effect for Buttons */
 .btn-group-toggle > .btn:hover, .btn-group-toggle > .btn.active:hover, .btn-group-toggle > .btn:focus {
     background-color: #ff5964 !important;
     color: #FFFFFF !important;
}
/* Remove Padding and Margin between Fluid Rows */
 .row > .col-sm-4, .row > .col-sm-3, .row > .col-sm-6 {
    /* padding: 0.0% !important;
     */
     padding: 0% !important;
     margin: 0 !important;
}
 .row {
     margin-left: 0 !important;
     margin-right: 0 !important;
}
/* Customize Button */
 .well .btn {
     white-space: normal !important;
     word-break: break-word;
}
 .btn.checkbtn.btn-custom {
     font-size: 14px !important;
     line-height: 1 !important;
}
/* Body Background Color */
 body {
     background-color: #ffffff;
}
 small {
     font-size: 16px;
}
/* Mobile Responsive Design */
 @media (max-width: 768px) {
     .dynamic-height {
         height: auto !important;
         min-height: 300px;
    }
     .col-sm-3, .col-sm-4, .col-sm-6 {
         width: 100% !important;
         margin-bottom: 5px;
    }
     .navbar-brand {
         font-size: 14px !important;
    }
     .btn {
         font-size: 14px !important;
         padding: 2px 2px !important;
    }
     .well, .wellPanel {
         padding: 1px !important;
         margin: 1px 0 !important;
    }
}
/* Show plot by default, hide the message */
 #sfPlotUnavailable {
     display: none;
}
 #networkUnavailable {
     display: none;
     height: 100%;
     justify-content: center;
     align-items: center;
     display: flex;
}
/* On small screens, hide plot and show the message */
 @media (max-width: 1000px), (max-height: 500px) {
     #sfPlotWrapper {
         display: none;
    }
     #sfPlotUnavailable {
         display: block;
    }
}
/* On small screens: hide the network output container, show the message */
 @media (max-width: 768px) {
     #mynetworkid {
         display: none !important;
    }
     #networkUnavailable {
         display: flex !important;
    }
}
 @media (max-width: 480px) {
     .dynamic-height {
         height: auto !important;
         min-height: 250px;
    }
     .navbar-brand {
         font-size: 14px !important;
    }
     .btn {
         font-size: 14px !important;
         padding: 4px 8px !important;
    }
}
    
}
    "))
  ),
  title = "Governance of Asia DB (GoA-DB)",
  theme = shinytheme("flatly"),
  navbarPage(id = "main_navbar",
             "🌏 Governance of Asia Database (GoA-DB)",
             tabPanel("Catalogue",
                      fluidPage(
                        fluidRow(
                          # Left panel for queries
                          column(3, div(class = "dynamic-height",
                                        wellPanel(
                                          
                                          div(
                                            style = "padding: 8px; margin-bottom: 0.5em;",
                                            div(
                                              style = "text-align: center;",
                                              h4(
                                                "Showing ",
                                                span(id = "filter_status_panel",
                                                     style = "border: 2px solid transparent; border-radius: 4px; padding: 0 4px; display: inline-block; text-align: center;",
                                                     textOutput("total_count", inline = TRUE)),
                                                " of ",
                                                textOutput("max_count", inline = TRUE),
                                                " individuals"
                                              )
                                              # ,
                                              # div(
                                              #   style = "font-size: 0.85em; color: #666; margin-top: -0.3em; margin-bottom: 0.4em;",
                                              #   "Plus ",
                                              #   textOutput("relatives_count", inline = TRUE),
                                              #   " relatives"
                                              # )
                                            ),
                                            div(
                                              id = "filter_buttons",
                                              style = "display: flex; gap: 10px; justify-content: center;",
                                              actionBttn(inputId = "reset_button", label = HTML("Reset"),
                                                         style = "simple", color = "default"),
                                              actionBttn(inputId = "go", label = HTML("Apply filter"),
                                                         style = "simple", color = "danger")
                                            ),
                                            div(
                                              id = "filter_loading",
                                              style = "display: none; margin-top: 8px; padding: 8px 12px; border: 1px solid #d9534f; border-radius: 4px; background: #fdf2f2; text-align: center; font-size: 0.9em; color: #a94442;",
                                              tags$span(class = "fa fa-spinner fa-spin", style = "margin-right: 6px;"),
                                              "Updating filters…"
                                            )
                                          ),
                                          
                                          # Label row with title + tooltip grouped tightly, toggle aligned right
                                          div(
                                            style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0.25em;",
                                            
                                            # Group title and tooltip together
                                            div(
                                              style = "display: flex; align-items: center; gap: 0.25em;",
                                              h4("Name of individual"),
                                            ),
                                            
                                            # Toggle with label, nudged up for baseline alignment
                                            div(
                                              style = "margin-top: 15px;",
                                              radioButtons(
                                                inputId = "query_name_logic_radio",
                                                label = NULL,
                                                inline = TRUE,
                                                choices = c("Any keyword" = "or", "All" = "and"),
                                                selected = "and"
                                              )
                                            )
                                          ),
                                          
                                          # Text input below
                                          div(
                                            style = "margin-top: -5px;",  # pulls input closer to the header group
                                            textInput("query_name", label = NULL, value = "", placeholder = "Hepburn, pinyin, or 漢字 (or leave blank)")
                                          ),
                                          
                                          div(
                                            style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0.25em;",
                                            # Group title and tooltip together
                                            div(
                                              style = "display: flex; align-items: center; gap: 0.25em;",
                                              h4("Year of birth")
                                            ),
                                            # Toggle with label, nudged up for baseline alignment
                                            div(
                                              style = "margin-top: 5.5px;",
                                              checkboxInput("include_unknown_birthyear", "Include unknown", value = TRUE)
                                            )
                                          ),
                                          
                                          div(style = "margin-top: -20px;",
                                              sliderInput(
                                                inputId = "time_birth",
                                                label = "",
                                                sep="",
                                                min=1845,
                                                max=1945,
                                                value = c(1845,1945),
                                                step = 1)
                                              # ),
                                          ),
                                          
                                          div(
                                            style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0.25em;",
                                            # Group title and tooltip together
                                            div(
                                              style = "display: flex; align-items: center; gap: 0.25em;",
                                              h4("Linguistic community")
                                            )
                                          ),

                                          div(style = "margin-top: -5px;",
                                              pickerInput(
                                                inputId = "languages",
                                                label = NULL,
                                                choices = c("⛩️️ Japanese" = "ja",
                                                            "🐉 Chinese" = "zh",
                                                            "🏯️️ Korean" = "kr",
                                                            "❓️️ Other" = "other"),
                                                selected = c("ja","zh","kr","other"),
                                                multiple = TRUE,
                                                options = pickerOptions(
                                                  actionsBox = TRUE,
                                                  selectedTextFormat = "count > 2",
                                                  countSelectedText = "{0} communities selected",
                                                  noneSelectedText = "Filter by community..."
                                                )
                                              )),
                                          
                                          
                                          # Label row with title + tooltip grouped tightly, toggle aligned right
                                          div(
                                            style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0.25em;",
                                            
                                            # Group title and tooltip together
                                            div(
                                              style = "display: flex; align-items: center; gap: 0.25em;",
                                              h4("Place of activity")
                                            ),
                                            
                                            # Toggle with label, nudged up for baseline alignment
                                            div(
                                              style = "margin-top: 15px;",
                                              radioButtons(
                                                inputId = "query_place_logic_radio",
                                                label = NULL,
                                                inline = TRUE,
                                                choices = c("Any keyword" = "or", "All" = "and"),
                                                selected = "or"
                                              )
                                            )
                                          ),
                                          
                                          div(style = "margin-top: -5px;",
                                              pickerInput(
                                                inputId = "place_countries",
                                                label = NULL,
                                                choices = c("⛩️ Japan" = "ja",
                                                            "🐉 China" = "zh",
                                                            "🏯️ Korea" = "kr",
                                                            "🏞️ Taiwan" = "tw",
                                                            "🌸 Manchukuo" = "mk"),
                                                selected = c("ja","zh","kr","tw","mk"),
                                                multiple = TRUE,
                                                options = pickerOptions(
                                                  actionsBox = TRUE,
                                                  selectedTextFormat = "count > 3",
                                                  countSelectedText = "{0} regions selected",
                                                  noneSelectedText = "Filter by region..."
                                                )
                                              )),

                                          div(
                                            style = "margin-top: -5px;",
                                            textInput("query_place", label = NULL, value = "", placeholder = "Hepburn, pinyin, or 漢字 (or leave blank)")
                                          ),

                                          div(
                                            style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0.25em;",
                                            # Group title and tooltip together
                                            div(
                                              style = "display: flex; align-items: center; gap: 0.25em;",
                                              h4("Year of activity")
                                            ),
                                            # Toggle with label, nudged up for baseline alignment
                                            div(
                                              style = "margin-top: 5.5px;",
                                              checkboxInput("include_unknown_year", "Include unknown", value = TRUE)
                                            )
                                          ),
                                          
                                          # div(h4("Year of graduation", style = "margin: 0;")),
                                          div(style = "margin-top: -20px;",
                                              sliderInput(
                                                inputId = "time_activity",
                                                label = "",
                                                sep="",
                                                min=1845,
                                                max=1945,
                                                value = c(1845,1945),
                                                step = 1)
                                          ),
                                          # # Label row with title + tooltip grouped tightly, toggle aligned right
                                          # div(
                                          #   style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0.25em;",
                                          #   
                                          #   # Group title and tooltip together
                                          #   div(
                                          #     style = "display: flex; align-items: center; gap: 0.25em;",
                                          #     h4("Colleges")
                                          #   ),
                                          #   
                                          #   # Toggle with label, nudged up for baseline alignment
                                          #   div(
                                          #     style = "margin-top: 15px;",
                                          #     radioButtons(
                                          #       inputId = "query_college_logic_radio",
                                          #       label = NULL,
                                          #       inline = TRUE,
                                          #       choices = c("Any keyword" = "or", "All" = "and"),
                                          #       selected = "and"
                                          #     )
                                          #   )
                                          # ),
                                          # 
                                          # # Text input below
                                          # div(
                                          #   style = "margin-top: -5px;",  # pulls input closer to the header group
                                          #   textInput("query_college", label = NULL, value = "", placeholder = "English or 漢字 (or leave blank)")
                                          # ),
                                          
                                          # Label row with title + tooltip grouped tightly, toggle aligned right
                                          div(
                                            style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0.25em;",
                                            
                                            # Group title and tooltip together
                                            div(
                                              style = "display: flex; align-items: center; gap: 0.25em;",
                                              h4("Employment")
                                            ),
                                            
                                            # Toggle with label, nudged up for baseline alignment
                                            div(
                                              style = "margin-top: 15px;",
                                              radioButtons(
                                                inputId = "query_employer_logic_radio",
                                                label = NULL,
                                                inline = TRUE,
                                                choices = c("Any keyword" = "or", "All" = "and"),
                                                selected = "and"
                                              )
                                            )
                                          ),
                                          
                                          # Text input below
                                          div(
                                            style = "margin-top: -5px;",  # pulls input closer to the header group
                                            textInput("query_employer", label = NULL, value = "", placeholder = "English or 漢字 (or leave blank)")
                                          ),

                                          div(style = "margin-top: -5px;",
                                              pickerInput(
                                                inputId = "hisco_filter",
                                                label = NULL,
                                                choices = c("Administrative/Managerial" = "2",
                                                            "Agricultural/Forestry/Fishing" = "6",
                                                            "Office/Admin Staff" = "3",
                                                            "Production/Transport" = "7",
                                                            "Professional/Technical" = "0",
                                                            "Sales" = "4",
                                                            "Service" = "5"),
                                                selected = NULL,
                                                multiple = TRUE,
                                                options = pickerOptions(actionsBox = TRUE,
                                                                        selectedTextFormat = "count > 3",
                                                                        countSelectedText = "{0} HISCO groups",
                                                                        noneSelectedText = "Filter by HISCO...")
                                              )
                                          ),

                                          div(style = "margin-top: -5px;",
                                              pickerInput(
                                                inputId = "isic_filter",
                                                label = NULL,
                                                choices = c("Accommodation and food service" = "I",
                                                            "Administrative and support service" = "N",
                                                            "Agriculture, forestry and fishing" = "A",
                                                            "Arts, entertainment and recreation" = "R",
                                                            "Construction" = "F",
                                                            "Education" = "P",
                                                            "Electricity, gas, steam and AC supply" = "D",
                                                            "Extraterritorial organizations" = "U",
                                                            "Financial and insurance" = "K",
                                                            "Health and social work" = "Q",
                                                            "Information and communication" = "J",
                                                            "Manufacturing" = "C",
                                                            "Mining and quarrying" = "B",
                                                            "Professional, scientific, technical" = "M",
                                                            "Public administration and defence" = "O",
                                                            "Real estate" = "L",
                                                            "Transportation and storage" = "H",
                                                            "Water supply, waste management" = "E",
                                                            "Wholesale and retail trade" = "G",
                                                            "Other service activities" = "S"),
                                                selected = NULL,
                                                multiple = TRUE,
                                                options = pickerOptions(actionsBox = TRUE,
                                                                        selectedTextFormat = "count > 3",
                                                                        countSelectedText = "{0} ISIC sectors",
                                                                        noneSelectedText = "Filter by ISIC...")
                                              )
                                          ),

                                          div(
                                            style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0.25em;",
                                            # Group title and tooltip together
                                            div(
                                              style = "display: flex; align-items: center; gap: 0.25em;",
                                              h4("Religions")
                                            )
                                          ),

                                          div(style = "margin-top: -5px;",
                                              pickerInput(
                                                inputId = "religion",
                                                label = NULL,
                                                choices = c("📿️️ Pure Land" = "pureland",
                                                            "🪷️ Zen" = "zen",
                                                            "☀️ Nichiren" = "nichiren",
                                                            "☸️️ Buddhism (other)" = "buddhism",
                                                            "⛩️️ Shintō" = "shinto",
                                                            "✝️️️ Christianity" = "christianity",
                                                            "📜️ Confucianism" = "confucianism",
                                                            "🧘 Tenrikyō" = "tenrikyo",
                                                            "❓️️ Other" = "other"),
                                                selected = c("pureland","zen","nichiren","buddhism","shinto","christianity","tenrikyo","confucianism","other"),
                                                multiple = TRUE,
                                                options = pickerOptions(
                                                  actionsBox = TRUE,
                                                  selectedTextFormat = "count > 3",
                                                  countSelectedText = "{0} religions selected",
                                                  noneSelectedText = "Filter by religion..."
                                                )
                                              )),
                                          
                                          div(
                                            style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0.25em;",
                                            div(
                                              style = "display: flex; align-items: center; gap: 0.3em;",
                                              h4("Genders")
                                            )
                                          ),

                                          div(style = "margin-top: -5px;",
                                              pickerInput(
                                                inputId = "genders",
                                                label = NULL,
                                                choices = c("♂ Male" = "m",
                                                            "♀ Female" = "f",
                                                            "◆️ Unknown" = "x"),
                                                selected = c("m","f","x"),
                                                multiple = TRUE,
                                                options = pickerOptions(
                                                  actionsBox = TRUE,
                                                  selectedTextFormat = "count > 2",
                                                  countSelectedText = "{0} genders selected",
                                                  noneSelectedText = "Filter by gender..."
                                                )
                                              )),
                                          
                                          div(
                                            style = "display: flex; align-items: center; justify-content: start; gap: 0.5em; margin-bottom: 0em;",
                                            # Group title and tooltip together
                                            div(
                                              style = "display: flex; align-items: center; gap: 0.3em;",
                                              h4("Search by ID")
                                            ),
                                            
                                            div(
                                              style = "margin-top: 6px;",  # pulls input closer to the header group
                                              textInput("query_id", label = NULL, value = "", placeholder = "")
                                            )
                                          ),
                                          
                                          tags$script(HTML("
                                            $(document).on('keyup', function(e) {
                                            if (e.which == 13 && $(e.target).is('#query_name, #query_place, #query_college, #query_employer, #query_id')) {  // Check if Enter is released inside the textInput
                                            $('#go').click();
                                            }
                                            });
                                          "))
                                        ))
                          ),
                          
                          column(9, 
                                 tabsetPanel(id = "main_tabs",
                                             tabPanel("People", value="People", 
                                                      column(7,
                                                             div(class = "center-content scrollable-content",
                                                                 wellPanel(class = "center-content  scrollable-content",
                                                                           withSpinner(DT::dataTableOutput("dataTablePerson", width = "100%"))
                                                                 ))),
                                                      column(5, div(class = "center-content",
                                                                    wellPanel(div(style="text-align: center",
                                                                                  p(downloadButton("download_excel_person", "Save selection (Excel)"),
                                                                                    downloadButton("download_csv_person", "Save selection (CSV)")))
                                                                    ),
                                                                    wellPanel(
                                                                      uiOutput("dynamic_ui_person")
                                                                    )
                                                      )
                                                      ),
                                             ),
                                             tabPanel("Colleges", value = "Colleges",
                                                      column(8,
                                                             div(class = "center-content scrollable-content",
                                                                 wellPanel(class = "center-content scrollable-content",
                                                                           withSpinner(DT::dataTableOutput("dataTableCol", width = "100%"))
                                                                 ))),
                                                      column(4, div(class = "center-content",
                                                                    wellPanel(div(style="text-align: center",
                                                                                  p(downloadButton("download_excel_col", "Save selection (Excel)"),
                                                                                    downloadButton("download_csv_col", "Save selection (CSV)")))
                                                                    ),wellPanel(
                                                                      uiOutput("dynamic_ui_cols")
                                                                    )
                                                      )
                                                      ),
                                             ),
                                             tabPanel("Employers", value = "Employers",
                                                      column(8,
                                                             div(class = "center-content scrollable-content",
                                                                 wellPanel(class = "center-content scrollable-content",
                                                                           withSpinner(DT::dataTableOutput("dataTablePub", width = "100%"))
                                                                 ))),
                                                      column(4, div(class = "center-content",
                                                                    wellPanel(div(style="text-align: center",
                                                                                  p(downloadButton("download_excel_pub", "Save selection (Excel)"),
                                                                                    downloadButton("download_csv_pub", "Save selection (CSV)")))
                                                                    ),wellPanel(
                                                                      uiOutput("dynamic_ui_pubs")
                                                                    )
                                                      )
                                                      ),
                                             ),
                                             tabPanel("Maps", div(class = "center-content full-width-plot",
                                                                  fluidRow(
                                                                    column(6,
                                                                           selectInput("asia_zoom", "Zoom on Region:",
                                                                                       choices = c("All East Asia", "East China", "Korea", "Manchuria", "South China", "Taiwan"),
                                                                                       selected = "All East Asia"),
                                                                           div(class = "center-content scrollable-content",
                                                                               wellPanel(
                                                                                 withSpinner(plotOutput("barChart_region_asia", width = "100%", height = "80vh"))
                                                                               )
                                                                           )
                                                                    ),
                                                                    column(6,
                                                                           selectInput("japan_zoom", "Zoom on Region:",
                                                                                       choices = c("All Japan", "Chubu", "Chugoku/Shikoku", "Hokkaido", "Kansai", "Kanto", "Kyushu", "Okinawa", "Tohoku"),
                                                                                       selected = "All Japan"),
                                                                           div(class = "center-content scrollable-content",
                                                                               wellPanel(
                                                                                 withSpinner(plotOutput("barChart_region_japan", width = "100%", height = "80vh"))
                                                                               )
                                                                           )
                                                                    )
                                                                  ),
                                                                  # fluidRow(
                                                                  #   column(12,
                                                                  #          div(class = "center-content scrollable-content",
                                                                  #              wellPanel(
                                                                  #                withSpinner(plotOutput("barChart_region", width = "100%", height = "80vh"))
                                                                  #              )
                                                                  #          )
                                                                  #   ),
                                                                  # )
                                                                  
                                             )),
                                             tabPanel(
                                               "Network",
                                               column(
                                                 8,
                                                 div(
                                                   class = "center-content",
                                                   wellPanel(
                                                     class = "dynamic-height-network center-content",
                                                     
                                                     visNetworkOutput("mynetworkid", height = "95%"),
                                                     div(
                                                       id = "networkUnavailable",
                                                       style = "display: none; text-align: center; font-weight: bold;",
                                                       "Network graph is not available on small display resolutions."
                                                     )
                                                   )
                                                 )
                                               ),
                                               column(
                                                 4,
                                                 div(
                                                   class = "center-content",
                                                   wellPanel(
                                                     uiOutput("dynamic_ui_netzwerk")
                                                   ),
                                                   wellPanel(
                                                     uiOutput("network_node_detail")
                                                   )
                                                 )
                                               )
                                             ),
                                             
                                             tabPanel("Statistics",
                                                      # First row
                                                      fluidRow(
                                                        column(6,
                                                               div(class = "center-content scrollable-content",
                                                                   wellPanel(
                                                                     withSpinner(plotOutput("barChart_birthyear", width = "100%", height = "38vh"))
                                                                   )
                                                               )
                                                        ),
                                                        column(6,
                                                               div(class = "center-content scrollable-content",
                                                                   wellPanel(
                                                                     withSpinner(plotOutput("barChart_fields", width = "100%", height = "38vh"))
                                                                   )
                                                               )
                                                        )
                                                      ),
                                                      fluidRow(
                                                        column(6,
                                                               div(class = "center-content scrollable-content",
                                                                   wellPanel(
                                                                     withSpinner(plotOutput("barChart_jobs", width = "100%", height = "38vh"))
                                                                   )
                                                               )
                                                        ),
                                                        column(6,
                                                               div(class = "center-content scrollable-content",
                                                                   wellPanel(
                                                                     withSpinner(plotOutput("barChart_activity", width = "100%", height = "38vh"))
                                                                   )
                                                               )
                                                        )
                                                      ),
                                                      fluidRow(
                                                        column(12,
                                                               div(class = "center-content scrollable-content",
                                                                   wellPanel(
                                                                     withSpinner(plotOutput("treemap_isic", width = "100%", height = "50vh"))
                                                                   )
                                                               )
                                                        )
                                                      ),

                                             )
                                 ),
                                 
                          ),
                        ),
                      )),
             
             tabPanel("Documentation",

                      fluidRow(
                        column(6, div(class = "dynamic-height scrollable-content", style = "padding-right: 8px;",
                                      wellPanel(tags$small(HTML(
                                        "<p><h4>About this database</h4></p>
                                         <p>The <b>Governance of Asia Database (GoA-DB)</b> is a historical prosopographical resource on individuals active in Japanese-controlled Asia between the late 19th century and 1945. Its records are mined from the multi-volume <i>Taishū jinji-roku</i> 大衆人事録 (\"Records of Public Individuals\") directories, then cleaned and linked using AI-assisted pipelines.</p>
                                         <p>It catalogues persons together with their birth years, places of origin, religions, ranks, hobbies, education, employment, and family relations, and ties each entry to organisations (employers, schools), occupations (HISCO/ISIC), and geolocated places across Japan, Korea, Taiwan, Manchukuo, and mainland China.</p>
                                         <p>Use the <b>Catalogue</b> tab to query and filter the dataset by name, place, era, gender, language, religion, occupation, employer, college, or ID; browse the matching individuals, employers, and colleges in the right-hand tables; and explore them spatially on the embedded maps, structurally as social networks, or longitudinally through the activity charts. Click any entry, employer, college, or location to inspect its detail panel and pivot the query around it.</p>
                                         <p style=\"font-size: 0.85em; color: #666; border-top: 1px solid #ddd; padding-top: 0.5em;\"><em>Disclaimer: the political entities, borders, and administrative units represented in this database (e.g. Manchukuo, the Japanese Empire, colonial Korea and Taiwan) reflect the historical situation of the period under study. Their inclusion is descriptive and carries no political judgment about the present.</em></p>"
                                      )))
                        )),
                        column(6, div(class = "dynamic-height scrollable-content", style = "padding-left: 8px;",
                                      wellPanel(tags$small(HTML(paste0(
                                        "<p><h4>Raw data download</h4></p>
                                         <p>The raw data will be available in a repository upon proper release of this database.</p>
                                         <p><h4>Presentations</h4></p>
                                         <p><ul><li>“The Japanese Governance of Asia: AI-Assisted Approaches to Mining, Cleaning, and Structuring Historical Prosopographical Data.” New Horizons for AI Research in the Social Sciences and Humanities, UChicago HK Campus, 2026.</li>
<li>“Imperial Governance in the <i>Taishū jinji-roku</i>: Data Mining the National Diet’s Next Digital Library.” HK Association for Digital Humanities Conference, Chinese University HK, 2026.</li></ul>
</p>",
                                        # "<p><h4>Primary sources</h4></p>
                                        #   <ul>
                                        #   <li>Imperial Secret Detective Authority 帝国秘密探偵社, ed. (1927): <i>Taishū jinji-roku Shōwa 3-nenban</i> 大衆人事録昭和3年版. doi: <a href=\"https://doi.org/10.11501/1688498\">10.11501/1688498</a>.</li>
                                        #   <li>⸻, ed. (1930): <i>Taishū jinji-roku dai 3-ban</i> 大衆人事録第3版. doi: <a href=\"https://doi.org/10.11501/3044845\">10.11501/3044845</a>.</li>
                                        #  <li>⸻, ed. (1932): <i>Taishū jinji-roku dai 5 (Shōwa 7-nen)-ban a-so no bu</i> 大衆人事録第5(昭和7年)版ア-ソ之部. doi: <a href=\"https://doi.org/10.11501/1688499\">10.11501/1688499</a>.</li>
                                        #  <li>⸻, ed. (1932): <i>Taishū jinji-roku dai 5 (Shōwa 7-nen)-ban ta-wa no bu</i> 大衆人事録第5(昭和7年)版タ-ワ之部. doi: <a href=\"https://doi.org/10.11501/1688500\">10.11501/1688500</a>.</li>
                                        #  <li>⸻, ed. (1935): <i>Taishū jinji-roku dai 11-ban</i> 大衆人事録第11版. doi: <a href=\"https://doi.org/10.11501/8312058\">10.11501/8312058</a>.</li>
                                        #  <li>⸻, ed. (1937): <i>Taishū jinji-roku [zenkoku-hen] 12-ban</i> 大衆人事録[全国篇]12版. doi: <a href=\"https://doi.org/10.11501/1686316\">10.11501/1686316</a>.</li>
                                        #  <li>⸻, ed. (1938): <i>Taishū jinji-roku dai 13-ban Tōkyō-hen</i> 大衆人事録第13版東京篇. doi: <a href=\"https://doi.org/10.11501/3017419\">10.11501/3017419</a>.</li>
                                        #  <li>⸻, ed. (1940): <i>Taishū jinji-roku Kantō Ōu Hokkaidō-hen</i> 大衆人事録関東・奥羽・北海道篇. doi: <a href=\"https://doi.org/10.11501/1173368\">10.11501/1173368</a>.</li>
                                        #  <li>⸻, ed. (1940): <i>Taishū jinji-roku Kinki-hen</i> 大衆人事録近畿篇. doi: <a href=\"https://doi.org/10.11501/1173383\">10.11501/1173383</a>.</li>
                                        #  <li>⸻, ed. (1940): <i>Taishū jinji-roku dai 13-ban Chūbu-hen</i> 大衆人事録第13版中部篇. doi: <a href=\"https://doi.org/10.11501/3017420\">10.11501/3017420</a>.</li>
                                        #  <li>⸻, ed. (1940): <i>Taishū jinji-roku dai 13-ban Chūgoku Shikoku Kyūshū-hen</i> 大衆人事録第13版中国・四国・九州篇. doi: <a href=\"https://doi.org/10.11501/1112914\">10.11501/1112914</a>.</li>
                                        #  <li>⸻, ed. (1940): <i>Taishū jinji-roku gaichi Man Shi kaigai-hen</i> 大衆人事録外地・満支・海外篇. doi: <a href=\"https://doi.org/10.11501/1173407\">10.11501/1173407</a>.</li>
                                        #  <li>⸻, ed. (1940): <i>Taishū jinji-roku dai 14-ban Hokkaidō Ōu Kantō Chūbu-hen</i> 大衆人事録第14版 北海道・奥羽・関東・中部篇. doi: <a href=\"https://doi.org/10.11501/1229896\">10.11501/1229896</a>.</li>
                                        #  <li>⸻, ed. (1942): <i>Taishū jinji-roku dai 14-ban Tōkyō-hen</i> 大衆人事録第14版 東京篇. doi: <a href=\"https://doi.org/10.11501/1683373\">10.11501/1683373</a>.</li>
                                        #  <li>⸻, ed. (1943): <i>Taishū jinji-roku dai 14-ban gaichi Man Shi kaigai-hen</i> 大衆人事録第14版 外地・満支・海外篇. doi: <a href=\"https://doi.org/10.11501/1230025\">10.11501/1230025</a>.</li>
                                        #  <li>⸻, ed. (1943): <i>Taishū jinji-roku dai 14-ban Kinki Chūgoku Shikoku Kyūshū-hen</i> 大衆人事録第14版 近畿・中国・四国・九州篇. doi: <a href=\"https://doi.org/10.11501/1229971\">10.11501/1229971</a>.</li>
                                        #  </ul></p>"
                                        "<p><h4>Primary sources</h4></p>
                                          <ul>
                                          <li>Imperial Secret Detective Authority 帝国秘密探偵社, ed. (1927): <i>Taishū jinji-roku Shōwa 3-nenban</i> 大衆人事録昭和3年版. doi: <a href=\"https://doi.org/10.11501/1688498\">10.11501/1688498</a>.</li>
                                         <li>⸻, ed. (1935): <i>Taishū jinji-roku dai 11-ban</i> 大衆人事録第11版. doi: <a href=\"https://doi.org/10.11501/8312058\">10.11501/8312058</a>.</li>
                                         <li>⸻, ed. (1940): <i>Taishū jinji-roku dai 14-ban Hokkaidō Ōu Kantō Chūbu-hen</i> 大衆人事録第14版 北海道・奥羽・関東・中部篇. doi: <a href=\"https://doi.org/10.11501/1229896\">10.11501/1229896</a>.</li>
                                         <li>⸻, ed. (1942): <i>Taishū jinji-roku dai 14-ban Tōkyō-hen</i> 大衆人事録第14版 東京篇. doi: <a href=\"https://doi.org/10.11501/1683373\">10.11501/1683373</a>.</li>
                                         <li>⸻, ed. (1943): <i>Taishū jinji-roku dai 14-ban gaichi Man Shi kaigai-hen</i> 大衆人事録第14版 外地・満支・海外篇. doi: <a href=\"https://doi.org/10.11501/1230025\">10.11501/1230025</a>.</li>
                                         <li>⸻, ed. (1943): <i>Taishū jinji-roku dai 14-ban Kinki Chūgoku Shikoku Kyūshū-hen</i> 大衆人事録第14版 近畿・中国・四国・九州篇. doi: <a href=\"https://doi.org/10.11501/1229971\">10.11501/1229971</a>.</li>
                                         </ul></p>"
                                      )))),
                                      
                        ))
                      )
                      
             ),
             
             tabPanel("Credits",
                      
                      fluidRow(
                        column(3),
                        column(6, div(class = "dynamic-height scrollable-content", align="center",
                                      wellPanel(tags$small(HTML(paste0(
                                        "
                                  <p><h4>Author</h4></p>
                                 <p>[Anonymized for review]</p>

                                 <p><h4>Project contact</h4></p>
                                 <p>[Anonymized for review]</p>

                                  <p><h4>Software</h4></p>

                                      <p>
                                      Data mined using<br/>
                                    <a href=\"https://github.com/ultralytics/ultralytics\" target=\"_blank\">Ultralytics YOLOe / YOLO v26</a>,
                                    <a href=\"https://doi.org/10.48550/arXiv.2505.09388\" target=\"_blank\">Qwen 3.5</a>
                                      </p>

                                    <p>
                                    Data prepared with<br/>
                                    <a href=\"https://doi.org/10.48550/arXiv.2311.11001\" target=\"_blank\">gendec-DistilBERT</a>,
                                    <a href=\"https://www.geonames.org/\" target=\"_blank\">GeoNames</a>,
                                    <a href=\"https://sedac.ciesin.columbia.edu/data/set/mcgd-mapping-chinese-geodata\" target=\"_blank\">MCGD</a>,
                                    <a href=\"https://github.com/rskmoi/namedivider-python\" target=\"_blank\">namedivider-python</a>,
                                    <a href=\"https://github.com/miurahr/pykakasi\" target=\"_blank\">pykakasi</a>,
                                  <a href=\"https://shiny.posit.co/\" target=\"_blank\">Shiny</a>,
                                  <a href=\"https://doi.org/10.32614/CRAN.package.tmcn\" target=\"_blank\">tmcn</a>,
                                  <a href=\"https://doi.org/10.32614/CRAN.package.visNetwork\" target=\"_blank\">visNetwork</a>
                                    </p>
                                    
                                    <p>
                                      Modified basemaps orginally by<br/>
                                    <a href=\"https://www.virtualshanghai.net/Maps/Base?ID=2210\" target=\"_blank\">Christian Henriot & Pierre-Henri Dubois</a> (Chinese Mainland, <a href=\"https://creativecommons.org/public-domain/cc0/\">CC-0 license</a>)<br/>
                                    <a href=\"https://searchworks.stanford.edu/view/12276480\" target=\"_blank\">Robert J. Hijmans & GADM</a> (Japan & Korea, for educational use)<br/>
                                    <a href=\"https://github.com/dkaoster/taiwan-atlas\" target=\"_blank\">Ministry of the Interior & Daniel Kao</a> (Taiwan, <a href=\"https://github.com/dkaoster/taiwan-atlas/blob/master/LICENSE\">MIT license</a>)
                                      </p>

                                     <p>
                                      Primary sources photographed by<br/>
                                    <a href=\"https://lab.ndl.go.jp/service/tsugidigi/\" target=\"_blank\">National Diet Library Next Digital Library</a> (public domain)
                                      </p> 

                                  <p><h4>Citation</h4></p>
                                  <p>[Anonymized for review], ed. (2026). “Governance of Asia Database.”</p>
                                 <p>GoA-DB is released under the <a href=\"https://creativecommons.org/licenses/by/4.0/legalcode\" target=\"_blank\">Creative Commons Attribution 4.0 International</a> licence: Citation required.<br/>
                                 BibTeX file available <a href=\"citation.bib\"  download=\"citation.bib\">here</a>.</p>
                                 <p><h4>Hosted by</h4></p>
                                 <p>[Anonymized for review]</p>
                                 "
                                      )))),
                                      
                        )),
                        column(3)
                      )
                      
             )
             
  ),
  tags$footer(
    class = "footer",
    HTML("[Anonymized for review], ed. (2026). “Governance of Asia Database.”"
    )))

# ==============================================================================
# SERVER LOGIC
# ==============================================================================
server <- function(input, output, session) {
  
  # expanded_fields <- reactive({
  #   unlist(lapply(input$fields, function(x) {
  #     switch(x,
  #            civil = c("civil", "hydraulic"),
  #            mechanical = "mechanical",
  #            electrical = c("electric","telecomm","radio"),
  #            mining = "mining",
  #            chemical = c("chemical","chemistry"),
  #            textile = "textile",
  #            other = "other"
  #     )
  #   }))
  # })
  
  rv <- reactiveValues(reset_done = FALSE)
  # Reactive values for data filtering and state management
  filtered_persons <- reactiveVal(data.frame())
  # filtered_fields <- reactiveVal()
  filtered_locations <- reactiveVal()
  filtered_colleges <- reactiveVal()
  filtered_employers <- reactiveVal()
  filtered_degrees <- reactiveVal()
  filtered_jobs <- reactiveVal()
  filtered_religions <- reactiveVal()
  
  max_total_rows <- reactiveVal(NULL)
  selected_node <- reactiveVal(NULL)
  selected_node_col <- reactiveVal(NULL)
  selected_node_org <- reactiveVal(NULL)
  
  # User interaction tracking
  clicked_person <- reactiveVal()
  clicked_organization <- reactiveVal()
  has_started <- reactiveVal()
  has_refiltered <- reactiveVal()
  
  # Visualization data
  plotted_points_var <- reactiveVal()
  plotted_points_countries_var <- reactiveVal()
  network_edges_shared <- reactiveVal()
  network_nodes_shared <- reactiveVal()
  
  first_run <- reactiveVal(TRUE)
  filter_is_active <- reactiveVal(FALSE)

  observe({
    active <- filter_is_active()
    if (active) {
      runjs("$('#filter_status_panel').css('border-color', '#d9534f');")
      runjs("$('#reset_button').closest('.bttn').removeClass('bttn-default').addClass('bttn-danger');")
    } else {
      runjs("$('#filter_status_panel').css('border-color', 'transparent');")
      runjs("$('#reset_button').closest('.bttn').removeClass('bttn-danger').addClass('bttn-default');")
    }
  })
  
  observeEvent(input$reset_button, {
    # Reset text inputs
    updateTextInput(session, "query_name", value = "")
    updateTextInput(session, "query_place", value = "")
    updateTextInput(session, "query_religion", value = "")
    updateTextInput(session, "query_hobby", value = "")
    updateTextInput(session, "query_college", value = "")
    updateTextInput(session, "query_employer", value = "")
    updateTextInput(session, "query_id", value = "")
    
    # Reset sliders
    updateSliderInput(session, "time_birth", value = c(1845, 1945))
    updateSliderInput(session, "time_activity", value = c(1845, 1945))
    
    # Reset checkbox groups
    # updateCheckboxGroupButtons(session, "fields", selected = c("civil","mechanical","textile","chemical","electrical","other","mining"))
    updatePickerInput(session, "genders", selected = c("m","f","x"))
    updatePickerInput(session, "languages", selected = c("ja","zh","kr","other"))
    updatePickerInput(session, "place_countries", selected = c("ja","zh","kr","tw","mk"))
    
    # Reset single checkboxes / radios
    updateCheckboxInput(session, "include_unknown_birthyear", value = TRUE)
    updateCheckboxInput(session, "include_unknown_year", value = TRUE)
    updatePickerInput(session, "religion", selected = c("pureland","zen","nichiren","buddhism","shinto","christianity","tenrikyo","confucianism","other"))
    
    updateRadioButtons(session, "query_name_logic_radio", selected = "and")
    updateRadioButtons(session, "query_place_logic_radio", selected = "or")
    # updateRadioButtons(session, "query_college_logic_radio", selected = "and")
    updateRadioButtons(session, "query_employer_logic_radio", selected = "and")
    updatePickerInput(session, "hisco_filter", selected = character(0))
    updatePickerInput(session, "isic_filter", selected = character(0))

    rv$reset_done <- FALSE
    later::later(function() { rv$reset_done <- TRUE }, 0.3)
  })
  
  observeEvent(rv$reset_done, {
    req(rv$reset_done)
    filter_data()
    has_started("T")
    runjs("$('#filter_loading').hide();")
  })
  
  # Wherever you detect a node click (e.g., in observeEvent)
  observeEvent(input$dataTablePerson_rows_selected, {
    filtered_data <- filtered_persons()
    
    if (length(input$dataTablePerson_rows_selected) > 0) {
      clicked <- filtered_data %>%
        select(person_id, name_family_latin, name_given_latin, name, birthyear, location_id, gender, domain) %>%
        unique() %>%
        arrange(name_family_latin, name_given_latin, name, birthyear, location_id) %>%
        slice(input$dataTablePerson_rows_selected)
      
      selected_node(clicked)
    } else {
      selected_node(NULL)
    }
  })
  
  # Wherever you detect a node click (e.g., in observeEvent)
  observeEvent(input$dataTablePub_rows_selected, {
    filtered_data <- filtered_employers()
    
    if (length(input$dataTablePub_rows_selected) > 0) {
      clicked <- filtered_data %>%
        select(n, organization_id, organization, location_id) %>%
        unique() %>%
        arrange(-n, organization, location_id) %>%
        slice(input$dataTablePub_rows_selected)
      
      selected_node_org(clicked)
      
    } else {
      selected_node_org(NULL)
    }
  })
  
  # Wherever you detect a node click (e.g., in observeEvent)
  observeEvent(input$dataTableCol_rows_selected, {
    filtered_data <- filtered_colleges()
    
    if (length(input$dataTableCol_rows_selected) > 0) {
      clicked <- filtered_data %>%
        select(n, organization_id, organization, location_id) %>%
        unique() %>%
        arrange(-n, organization, location_id) %>%
        slice(input$dataTableCol_rows_selected)
      
      selected_node_col(clicked)
      
    } else {
      selected_node_col(NULL)
    }
  })
  
  make_network_nodes <- function() {
    filtered_persons() %>%
      mutate(id = paste0(person_id, "_person"), title = paste(name_family_latin,name_given_latin,name), label = "", group = "person") %>%
      select(group, title, id, label) %>%
      rbind(
        filtered_colleges() %>%
          mutate(id = paste0(organization_id, "_college"), title = organization, label = "", group = "college") %>%
          select(group, title, id, label) %>%
          unique()
      ) %>%
      rbind(
        filtered_employers() %>%
          mutate(id = paste0(organization_id, "_employer"), title = organization, label = "", group = "employer") %>%
          select(group, title, id, label)
      ) %>%
      rbind(
        records_locations %>%
          st_drop_geometry() %>%
          select(location_display, location_id) %>%
          filter(location_id %in% filtered_persons()$location_id) %>%
          mutate(id = paste0(location_id, "_location"), title = location_display, label = "", group = "location") %>%
          select(group, title, id, label)
      ) %>%
      unique() %>%
      group_by(id) %>%
      slice(1) %>%
      ungroup() %>%
      drop_na()
  }
  
  make_network_edges <- function(nodes_df) {
    filtered_persons() %>%
      rename(from = person_id, to = location_id) %>%
      select(from, to) %>%
      drop_na() %>%
      mutate(from = paste0(from, "_person"), to = paste0(to, "_location")) %>%
      rbind(
        filtered_degrees() %>%
          rename(from = person_id, to = organization_id) %>%
          select(from, to) %>%
          mutate(from = paste0(from, "_person"), to = paste0(to, "_college"))
      ) %>%
      rbind(
        filtered_jobs() %>%
          rename(from = person_id, to = organization_id) %>%
          select(from, to) %>%
          drop_na() %>%
          mutate(from = paste0(from, "_person"), to = paste0(to, "_employer"))
      ) %>%
      rbind(
        filtered_employers() %>%
          filter(!is.na(parent_organization_id)) %>%
          rename(from = organization_id, to = parent_organization_id) %>%
          select(from, to) %>%
          drop_na() %>%
          mutate(from = paste0(from, "_employer"), to = paste0(to, "_employer"))
      ) %>%
      drop_na() %>%
      filter(from != to, from %in% nodes_df$id, to %in% nodes_df$id) %>%
      unique()
  }
  
  # ------------------------------------------------------------------------------
  # Network Graph Output  
  # ------------------------------------------------------------------------------
  
  observeEvent(total_rows(), once = TRUE, {
    max_total_rows(total_rows())
  })

  output$total_count <- renderText({
    format(total_rows(), big.mark = ",", scientific = FALSE)
  })

  output$max_count <- renderText({
    req(max_total_rows())
    format(max_total_rows(), big.mark = ",", scientific = FALSE)
  })

  # output$relatives_count <- renderText({
  #   req(!is.null(filtered_persons()), "person_id" %in% names(filtered_persons()))
  #   pids <- filtered_persons()$person_id
  #   n_rel <- records_family %>%
  #     filter(person_id %in% pids) %>%
  #     filter(!if_all(-c(person_id, relation_id, order, order2, order3), is.na)) %>%
  #     nrow()
  #   format(n_rel, big.mark = ",", scientific = FALSE)
  # })
  
  output$mynetworkid <- renderVisNetwork({
    
    if (total_rows() >= 2000) {
      return(visNetwork(
        nodes = data.frame(id = "msg_large", label = "Network too large to compute (>2,000 individuals). Please limit your query."),
        edges = data.frame()
      ) %>%
        visOptions(highlightNearest = FALSE, nodesIdSelection = FALSE) %>%
        visInteraction(dragNodes = FALSE, dragView = FALSE, zoomView = FALSE))
    }
    
    g_subgraph <- g_sub()
    req(g_subgraph)
    
    if (vcount(g_subgraph) > 4000) {
      return(visNetwork(
        nodes = data.frame(id = "msg_dense", label = "Network computed, but too dense to visualise (>4,000 nodes). Please limit your query."),
        edges = data.frame()
      ) %>%
        visOptions(highlightNearest = FALSE, nodesIdSelection = FALSE) %>%
        visInteraction(dragNodes = FALSE, dragView = FALSE, zoomView = FALSE))
    }
    
    nodes <- network_nodes_shared() %>% filter(id %in% V(g_subgraph)$name)
    edges <- as_data_frame(g_subgraph, what = "edges")

    # Compute eigenvector centrality and map to gentle size scaling
    eig <- eigen_centrality(g_subgraph)$vector
    eig_df <- data.frame(id = names(eig), eig_c = as.numeric(eig), stringsAsFactors = FALSE)

    # Per-node icon size scaled by eigenvector centrality
    icon_codes  <- c(college = "f19c", degree = "f19d", person = "f183", employer = "f1ad", job = "f0b1", location = "f3c5")
    icon_colors <- c(college = "#9b0a7d", degree = "#9b0a7d", person = "#c34113", employer = "#428bca", job = "#428bca", location = "darkgreen")
    base_sizes  <- c(college = 25, degree = 25, person = 25, employer = 25, job = 25, location = 25)
    nodes <- nodes %>%
      left_join(eig_df, by = "id") %>%
      mutate(
        eig_c = ifelse(is.na(eig_c), 0, eig_c),
        shape = "icon",
        icon.code  = icon_codes[group],
        icon.color = icon_colors[group],
        icon.size  = as.integer(base_sizes[group] * (1 + 1.5 * log1p(eig_c * 100) / log1p(100)))
      ) %>%
      select(-eig_c)

    set.seed(1337)
    visNetwork(nodes, edges) %>%
      addFontAwesome() %>%
      visIgraphLayout(layout = "layout_with_fr") %>%
      visPhysics(stabilization = FALSE, timestep = .35, minVelocity = 10,
                 maxVelocity = 50, solver = "forceAtlas2Based") %>%
      visEdges(smooth = FALSE) %>%
      visOptions(highlightNearest = list(enabled = TRUE, degree = 2, hover = TRUE)) %>%
      visEvents(click = "function(nodes) {
      Shiny.onInputChange('clicked_node',
        nodes.nodes.length > 0 ? nodes.nodes[0] : null);
    }")
  })
  
  # Reactive observer to filter data — listens to button click
  observeEvent(input$go, {
    rv$reset_done <- FALSE
    later::later(function() { rv$reset_done <- TRUE }, 0.3)
    # filter_data()
    # has_started("T")
  })
  
  observeEvent(input$main_tabs, {
    if(is_null(has_started())==F & is_null(has_refiltered())==F & input$main_tabs!="Employers"){
      filter_data()
      has_refiltered(NULL)
    }
  })
  
  # --- Colleges ---
  observeEvent(input$clicked_college_id, {
    req(input$clicked_college_id)
    updateTextInput(session, "query_id", value = input$clicked_college_id)
    filter_data(input$clicked_college_id)
  }, ignoreInit = TRUE)

  # --- Employers ---
  observeEvent(input$clicked_employer_id, {
    req(input$clicked_employer_id)
    updateTextInput(session, "query_id", value = input$clicked_employer_id)
    filter_data(input$clicked_employer_id)
  }, ignoreInit = TRUE)
  
  # --- Subsidies (disabled: parent_id column no longer exists in data) ---
  # observeEvent(input$clicked_subsidy_id, {
  #   req(input$clicked_subsidy_id)
  #   updateTabsetPanel(session, "main_tabs", selected = "Employers")
  #   subsidies <- records_career %>% filter(parent_id %in% input$clicked_subsidy_id | organization_id %in% input$clicked_subsidy_id)
  #   filter_data(subsidies$organization_id)
  # }, ignoreInit = TRUE)
  
  # --- Entry filter ---
  observeEvent(input$clicked_entry_id, {
    req(input$clicked_entry_id)
    updateTabsetPanel(session, "main_tabs", selected = "People")
    updateTextInput(session, "query_id", value = input$clicked_entry_id)
    filter_data(input$clicked_entry_id)
  }, ignoreInit = TRUE)

  # --- Location link click: filter by exact location_id ---
  observeEvent(input$clicked_location_id, {
    req(input$clicked_location_id)
    updateTabsetPanel(session, "main_tabs", selected = "People")
    updateTextInput(session, "query_id", value = input$clicked_location_id)
    filter_data(input$clicked_location_id)
  }, ignoreInit = TRUE)

  # --- Network node: navigate to People tab and select row ---
  observeEvent(input$nav_person_id, {
    req(input$nav_person_id)
    updateTabsetPanel(session, "main_tabs", selected = "People")
    row_idx <- which(dataTablePerson()$person_id == input$nav_person_id)
    if (length(row_idx) > 0) {
      proxy <- DT::dataTableProxy("dataTablePerson")
      DT::selectRows(proxy, row_idx[1])
    }
  }, ignoreInit = TRUE)

  # --- Network node: navigate to Employers tab and select row ---
  observeEvent(input$nav_employer_id, {
    req(input$nav_employer_id)
    updateTabsetPanel(session, "main_tabs", selected = "Employers")
    tbl <- filtered_employers() %>% arrange(-n, organization, location_id)
    row_idx <- which(tbl$organization_id == input$nav_employer_id)
    if (length(row_idx) > 0) {
      proxy <- DT::dataTableProxy("dataTablePub")
      DT::selectRows(proxy, row_idx[1])
    }
  }, ignoreInit = TRUE)

  # --- Network node: navigate to Colleges tab and select row ---
  observeEvent(input$nav_college_id, {
    req(input$nav_college_id)
    updateTabsetPanel(session, "main_tabs", selected = "Colleges")
    tbl <- filtered_colleges() %>% arrange(-n, organization, location_id)
    row_idx <- which(tbl$organization_id == input$nav_college_id)
    if (length(row_idx) > 0) {
      proxy <- DT::dataTableProxy("dataTableCol")
      DT::selectRows(proxy, row_idx[1])
    }
  }, ignoreInit = TRUE)

  # --- Network: focus a node in the graph (from centrality links) ---
  observeEvent(input$focus_network_node, {
    req(input$focus_network_node)
    node_id <- input$focus_network_node
    visNetworkProxy("mynetworkid") %>%
      visFocus(id = node_id, scale = 1.5) %>%
      visSelectNodes(id = list(node_id))
    # Also trigger the detail panel
    shinyjs::runjs(paste0("Shiny.setInputValue('clicked_node', '", node_id, "', {priority: 'event'});"))
  }, ignoreInit = TRUE)

  # ------------------------------------------------------------------------------
  # Core Data Filtering Function
  # ------------------------------------------------------------------------------
  # This function applies all user-selected filters to the research data
  
  filter_data <- function(id="") {
    # req(input$fields)  # Make sure places are selected
    # req(input$datasets)
    
    query_name_logic <- input$query_name_logic_radio
    query_place_logic <- input$query_place_logic_radio
    query_employer_logic <- input$query_employer_logic_radio
    # query_college_logic <- input$query_college_logic_radio
    
    # query_college <- toTrad(input$query_college)

    if (length(id) == 0 || all(id == "")) {
      query_id <- trimws(if (is.null(input$query_id)) "" else input$query_id)

      # Parse user search query
      query_name <- toTrad(input$query_name)
      query_place <- toTrad(input$query_place)
      query_employer <- toTrad(input$query_employer)

    }else{
      query_id <- trimws(id)
      query_name <- NULL
      query_place <- ""
      query_employer <- ""
      updateTextInput(session, "query_name", value = "")
      updateTextInput(session, "query_employer", value = "")
      updateTextInput(session, "query_place", value = "")
      updatePickerInput(session, "hisco_filter", selected = character(0))
      updatePickerInput(session, "isic_filter", selected = character(0))
    }
    
    # Store selected faculty filters for use across reactive expressions
    
    selected_religions <- input$religion
    filtered_religions(selected_religions)
    
    # If query_id is a location ID (starts with "L"), filter locations by exact ID
    if (!is.null(query_id) && length(query_id) == 1 && nzchar(query_id) && startsWith(query_id, "L")) {
      filtered_locations(records_locations %>% filter(location_id == query_id))
    } else {
      filtered_locations(
        records_locations %>%
        {  # handle name query separately
          if (is.null(query_place) || query_place == "") {
            .
          } else {
            query_place_words <- strsplit(query_place, "\\s+")[[1]]

            word_matches <- lapply(query_place_words, function(word) {
              pattern <- paste0("\\b", word, "\\b")
              grepl(pattern, .$location_name, ignore.case = TRUE) |
                grepl(pattern, .$name_en, ignore.case = TRUE) |
                grepl(pattern, .$province, ignore.case = TRUE)
            })

            if (query_place_logic == "or") {
              filter(., Reduce(`|`, word_matches))
            } else if (query_place_logic == "and") {
              filter(., Reduce(`&`, word_matches))
            } else {
              stop("Query logic must be 'or' or 'and'")
            }
          }
        }
      )
    }
    
    filtered_degrees(
      records_education %>%
        # 
        # # filter({
        # #   # Use the expanded fields
        # #   selected_fields <- expanded_fields()
        # #   
        # #   if(length(selected_fields) == 0) {
        # #     TRUE
        # #   } else {
        # #     # Keep "other" separate for filtering
        # #     main_fields <- setdiff(selected_fields, "other")
        # #     main_pattern <- paste(main_fields, collapse = "|")
        # #     
        # #     if("other" %in% selected_fields & length(main_fields) == 0) {
        # #       # only "other" selected -> keep degrees not matching any known field
        # #       !grepl("civil|hydraulic|mechanical|electric|telecomm|radio|mining|chemical|chemistry|textile", field, ignore.case = TRUE)
        # #     } else if("other" %in% selected_fields) {
        # #       # "other" plus main fields -> keep main fields OR anything not matching known fields
        # #       grepl(main_pattern, field, ignore.case = TRUE) |
        # #         !grepl("civil|hydraulic|mechanical|electric|telecomm|radio|mining|chemical|chemistry|textile", field, ignore.case = TRUE)
        # #     } else {
        # #       # only main fields
        # #       grepl(main_pattern, field, ignore.case = TRUE)
        # #     }
        # #   }
        # # }) %>%
        # 
        # filter(
        #   year_graduated >= input$time_graduation[[1]] &
        #     year_graduated <= input$time_graduation[[2]] |
        #     (input$include_unknown_graduation & is.na(year_graduated))
        # ) %>%
        # 
        left_join(records_organizations %>% select(organization_id,organization), by="organization_id") %>%

        {

          if (!is.null(query_id) && length(query_id) == 1 && nzchar(query_id) && startsWith(query_id, "O")) {
            filter(., organization_id %in% query_id)
          } else {
            .
          }
        } %>%

        {  # handle name query separately
          if (is.null(query_employer) || query_employer == "") {
            .
          } else {
            query_employer_words <- strsplit(query_employer, "\\s+")[[1]]

            word_matches <- lapply(query_employer_words, function(word) {
              pattern <- paste0("\\b", word, "\\b")
              grepl(pattern, .$organization, ignore.case = TRUE)
            })

            if (query_employer_logic == "or") {
              filter(., Reduce(`|`, word_matches))
            } else if (query_employer_logic == "and") {
              filter(., Reduce(`&`, word_matches))
            } else {
              stop("Query logic must be 'or' or 'and'")
            }
          }
        }
    )

    filtered_jobs(
      records_career %>%

        left_join(records_organizations %>% select(organization_id,organization), by="organization_id") %>%

        {
          if (!is.null(query_id) && length(query_id) == 1 && nzchar(query_id) && startsWith(query_id, "O")) {
            filter(., organization_id %in% query_id)
          } else {
            .
          }
        } %>%
        
        {  # handle name query separately
          if (is.null(query_employer) || query_employer == "") {
            .
          } else {
            query_employer_words <- strsplit(query_employer, "\\s+")[[1]]
            
            word_matches <- lapply(query_employer_words, function(word) {
              pattern <- paste0("\\b", word, "\\b")
              grepl(pattern, .$organization, ignore.case = TRUE) |
                grepl(pattern, .$job_title, ignore.case = TRUE)
            })
            
            if (query_employer_logic == "or") {
              filter(., Reduce(`|`, word_matches))
            } else if (query_employer_logic == "and") {
              filter(., Reduce(`&`, word_matches))
            } else {
              stop("query_name_logic must be 'or' or 'and'")
            }
          }
        } %>%

        {  # HISCO filter
          hisco_sel <- input$hisco_filter
          if (is.null(hisco_sel) || length(hisco_sel) == 0) {
            .
          } else {
            # Map group codes to all matching hisco_major digits
            hisco_digits <- hisco_sel
            if ("7" %in% hisco_sel) hisco_digits <- c(hisco_digits, "8", "9")
            if ("0" %in% hisco_sel) hisco_digits <- c(hisco_digits, "1")
            filter(., as.character(hisco_major) %in% hisco_digits)
          }
        } %>%

        {  # ISIC filter
          isic_sel <- input$isic_filter
          if (is.null(isic_sel) || length(isic_sel) == 0) {
            .
          } else {
            org_ids <- records_organizations %>%
              filter(isic_section %in% isic_sel) %>%
              pull(organization_id)
            filter(., organization_id %in% org_ids)
          }
        }
    )
    
    filtered_persons(
      records_persons %>%

        filter(!is.na(name_family_latin) & nchar(name_family_latin) >= 2) %>%

        {
          if (!is.null(query_id) && length(query_id) == 1 && nzchar(query_id) && startsWith(query_id, "P")) {
            filter(., person_id %in% query_id)
          } else {
            .
          }
        } %>%
        
        # filter(person_id %in% (records_persons %>% filter(dataset %in% input$datasets))$person_id) %>%
        
        {
          sel_rel <- input$religion
          if (is.null(sel_rel) || length(sel_rel) == 0) {
            .
          } else {
            matching_rels <- religion_category_map %>%
              filter(picker_cat %in% sel_rel) %>%
              pull(religion) %>% unique()
            include_other <- "other" %in% sel_rel
            filter(., religion %in% matching_rels |
                     (include_other & (is.na(religion) | !religion %in% religion_category_map$religion)))
          }
        } %>%
        
        {
          # Default: no career/education requirement — many entries are valid
          # without either (minors, retirees, persons with only family/rank).
          # Narrow to job-having persons only when HISCO or ISIC is active,
          # since both are occupation-class filters the user opted into and
          # are evaluated against career records.
          isic_sel <- input$isic_filter
          hisco_sel <- input$hisco_filter
          job_filter_active <-
            (!is.null(isic_sel) && length(isic_sel) > 0) ||
            (!is.null(hisco_sel) && length(hisco_sel) > 0)
          if (job_filter_active) {
            filter(., person_id %in% filtered_jobs()$person_id)
          } else {
            .
          }
        } %>%
        
        filter(
          (location_id %in% filtered_locations()$location_id) |
            (is.na(location_id) &
               (is.null(query_place) || length(query_place) == 0 || all(query_place == "")) &
               !(length(query_id) == 1 && nzchar(query_id) && startsWith(query_id, "L")))
        ) %>%

        {
          sel_countries <- input$place_countries
          all_countries <- c("ja","zh","kr","tw","mk")
          if (is.null(sel_countries) || setequal(sel_countries, all_countries)) {
            .
          } else {
            pids_in_countries <- person_career_countries %>%
              filter(
                country %in% setdiff(sel_countries, "mk") |
                (("mk" %in% sel_countries) & province %in% manchukuo_provinces)
              ) %>%
              pull(person_id) %>% unique()
            filter(., person_id %in% pids_in_countries)
          }
        } %>%

        filter(tolower(gender) %in% tolower(input$genders)) %>%
        
        filter(tolower(domain) %in% tolower(input$languages)) %>%
        
        filter(
          birthyear >= input$time_birth[[1]] &
            birthyear <= input$time_birth[[2]] |
            (input$include_unknown_birthyear & is.na(birthyear))
        ) %>%
        
        {
          if (is.null(query_name) || query_name == "") {
            .
          } else {
            query_name_words <- strsplit(query_name, "\\s+")[[1]]
            
            word_matches <- lapply(query_name_words, function(word) {
              if (query_name_logic == "or") {
                # Allow partial matches (no word boundaries)
                pattern <- paste0(word)
              } else if (query_name_logic == "and") {
                # Require exact word boundaries
                pattern <- paste0("\\b", word, "\\b")
              } else {
                stop("Query logic must be 'or' or 'and'")
              }
              
              grepl(pattern, .$name, ignore.case = TRUE) |
                grepl(pattern, .$name_given, ignore.case = TRUE) |
                grepl(pattern, .$name_given_latin, ignore.case = TRUE) |
                grepl(pattern, .$name_family, ignore.case = TRUE) |
                grepl(pattern, .$name_family_latin, ignore.case = TRUE)
            })
            
            if (query_name_logic == "or") {
              filter(., Reduce(`|`, word_matches))
            } else if (query_name_logic == "and") {
              filter(., Reduce(`&`, word_matches))
            } else {
              stop("Query logic must be 'or' or 'and'")
            }
          }
        }
    )
    
    filtered_jobs(
      filtered_jobs() %>%
        filter(person_id %in% filtered_persons()$person_id)
    )
    
    filtered_degrees(
      filtered_degrees() %>%
        filter(person_id %in% filtered_persons()$person_id)
    )
    
    filtered_colleges({
      base_counts <- filtered_degrees() %>%
        select(person_id, organization_id) %>%
        unique() %>%
        count(organization_id)
      # Add subdivision student counts to parent orgs (mirrors filtered_employers)
      org_parents <- records_organizations %>%
        select(organization_id, parent_organization_id) %>%
        filter(!is.na(parent_organization_id))
      child_sums <- base_counts %>%
        inner_join(org_parents, by = "organization_id") %>%
        group_by(parent_organization_id) %>%
        summarise(child_n = sum(n), .groups = "drop") %>%
        rename(organization_id = parent_organization_id)
      base_counts %>%
        left_join(child_sums, by = "organization_id") %>%
        mutate(n = n + coalesce(child_n, 0L)) %>%
        select(-child_n) %>%
        left_join(records_organizations %>% select(organization_id, organization, location_id, parent_organization_id), by = "organization_id") %>%
        filter(!organization == "") %>% filter(!is.na(organization)) %>% filter(!organization == "NA")
    })
    
    filtered_employers({
      base_counts <- filtered_jobs() %>%
        select(person_id, organization_id) %>%
        unique() %>%
        count(organization_id)
      # Add subdivision counts to parent orgs
      org_parents <- records_organizations %>%
        select(organization_id, parent_organization_id) %>%
        filter(!is.na(parent_organization_id))
      child_sums <- base_counts %>%
        inner_join(org_parents, by = "organization_id") %>%
        group_by(parent_organization_id) %>%
        summarise(child_n = sum(n), .groups = "drop") %>%
        rename(organization_id = parent_organization_id)
      base_counts %>%
        left_join(child_sums, by = "organization_id") %>%
        mutate(n = n + coalesce(child_n, 0L)) %>%
        select(-child_n) %>%
        left_join(records_organizations %>% select(organization_id, organization, location_id, parent_organization_id), by = "organization_id") %>%
        filter(!organization == "")
    })
    
    plotted_points <- records_persons %>% 
      filter(location_id %in% filtered_persons()$location_id) %>%
      left_join(records_locations,by="location_id",relationship="many-to-many") %>%
      select(person_id,geometry) %>%
      unique() %>%
      count(geometry) %>%
      arrange(-n)
    
    plotted_points_var(plotted_points)

    # Detect whether any filter is active (differs from defaults)
    any_active <- FALSE
    if (!is.null(input$query_name) && nzchar(input$query_name)) any_active <- TRUE
    if (!is.null(input$query_place) && nzchar(input$query_place)) any_active <- TRUE
    if (!is.null(input$query_employer) && nzchar(input$query_employer)) any_active <- TRUE
    if (!is.null(input$query_id) && nzchar(input$query_id)) any_active <- TRUE
    if (!is.null(input$query_hobby) && nzchar(input$query_hobby)) any_active <- TRUE
    if (!is.null(input$query_college) && nzchar(input$query_college)) any_active <- TRUE
    if (!setequal(input$genders, c("m","f","x"))) any_active <- TRUE
    if (!setequal(input$languages, c("ja","zh","kr","other"))) any_active <- TRUE
    if (!setequal(input$place_countries, c("ja","zh","kr","tw","mk"))) any_active <- TRUE
    if (!setequal(input$religion, c("pureland","zen","nichiren","buddhism","shinto","christianity","tenrikyo","confucianism","other"))) any_active <- TRUE
    if (!identical(as.numeric(input$time_birth), c(1845, 1945))) any_active <- TRUE
    if (!isTRUE(input$include_unknown_birthyear)) any_active <- TRUE
    if (length(input$hisco_filter) > 0) any_active <- TRUE
    if (length(input$isic_filter) > 0) any_active <- TRUE
    filter_is_active(any_active)

  }
  
  # --- Reactive nodes / edges ---
  network_nodes_shared <- reactive({
    req(total_rows() < 2000)
    make_network_nodes()
  })
  
  network_edges_shared <- reactive({
    req(total_rows() < 2000)
    make_network_edges(network_nodes_shared())
  })
  
  # --- Full graph ---
  g <- reactive({
    req(total_rows() < 2000)
    edges <- network_edges_shared()
    req(nrow(edges) > 0)
    
    graph_from_edgelist(as.matrix(edges[, c("from", "to")]), directed = FALSE)
  })
  
  # --- Subgraph ---
  g_sub <- reactive({
    full_graph <- g()
    comps <- components(full_graph)
    induced_subgraph(
      full_graph,
      V(full_graph)[comps$membership %in%
                      which(comps$csize >= 0.05 * max(comps$csize))]
    )
  })
  
  total_rows <- reactive({
    req(!is.null(filtered_persons()), "person_id" %in% names(filtered_persons()))
    nrow(filtered_persons() %>% select(person_id) %>% distinct())
  })
  
  # ------------------------------------------------------------------------------
  # Logics First Startup
  # ------------------------------------------------------------------------------
  
  # --- Trigger manually once on startup ---
  observeEvent(TRUE, {
    if (!is.null(input$religion)) {
      filter_data()
      has_started("T")
    }
  }, once = TRUE)
  
  # ------------------------------------------------------------------------------
  # Data Table: People
  # ------------------------------------------------------------------------------
  
  # One canonical reactive for the table displayed in the UI
  dataTablePerson <- reactive({
    base <- filtered_persons() %>%
      select(person_id, name_family_latin, name_given_latin, name, birthyear, location_id, gender, domain) %>%
      unique()

    pids <- unique(base$person_id)

    # Latest job location per person, with org-level fallback
    careers <- records_career %>%
      filter(person_id %in% pids)
    if (nrow(careers) > 0) {
      org_loc_lookup <- records_organizations %>%
        filter(organization_id %in% careers$organization_id) %>%
        select(organization_id, location_id) %>%
        distinct(organization_id, .keep_all = TRUE) %>%
        deframe()
      careers <- careers %>%
        mutate(effective_location_id = coalesce(
          location_id,
          unname(org_loc_lookup[organization_id])
        ))
    } else {
      careers$effective_location_id <- character(0)
    }
    latest_job <- careers %>%
      filter(!is.na(effective_location_id)) %>%
      group_by(person_id) %>%
      arrange(
        desc(!is.na(start_year)),
        desc(start_year),
        desc(current %in% TRUE),
        .by_group = TRUE
      ) %>%
      slice(1) %>%
      ungroup() %>%
      select(person_id, latest_loc = effective_location_id)

    base %>%
      left_join(latest_job, by = "person_id") %>%
      # Use the latest-residence (latest job loc → native loc fallback) so the
      # "Latest residence" column resolves even when the person's own
      # location_id is null but a job's organization has a known location.
      mutate(location_id = coalesce(latest_loc, location_id)) %>%
      select(-latest_loc) %>%
      left_join(
        records_locations %>% st_drop_geometry() %>%
          select(location_id, location_display_short),
        by = "location_id", relationship = "many-to-many"
      ) %>%
      select(person_id, name_family_latin, name_given_latin, name, birthyear,
             location_display_short, location_id, gender, domain) %>%
      unique() %>%
      arrange(name_family_latin, name_given_latin, name, birthyear, location_display_short)
  })
  #
  # dataTablePerson <- reactive({
  #   filtered_persons()
  # })

  # dataTablePerson <-reactive({
  #   records_persons %>%
  #     select(person_id, name_family_latin, name_given_latin, name, birthyear, location_id)  %>%
  #     unique() %>%
  #     left_join(records_locations, by = "location_id", relationship = "many-to-many") %>%
  #     select(person_id, name_family_latin, name_given_latin, name, birthyear, location_name, location_id)  %>%
  #     unique() %>%
  #     arrange(name_family_latin, name_given_latin, name, birthyear, location_name)
  #   })

  # Render it
  output$dataTablePerson <- DT::renderDataTable({
    DT::datatable(
      dataTablePerson() %>%
        select(name_family_latin,name_given_latin,name,birthyear,location_display_short) %>%
        rename('Family name' = name_family_latin, 'Given name' = name_given_latin, 'Birth year' = birthyear,
               'Latest residence' = location_display_short, Characters = name),
      selection = list(mode = "single"),
      rownames = FALSE,
      options = list(
        orderClasses = TRUE,
        pageLength = 25,
        dom = 'tip'
      )
    )
  })
  
  # ------------------------------------------------------------------------------
  # Data Table: Employers
  # ------------------------------------------------------------------------------
  
  dataTablePub <- reactive ({
    datatable_employers <- filtered_employers()
    
    # Check if the filtered data is empty
    if (is.null(datatable_employers) || nrow(datatable_employers) == 0) {
      return(data.frame("No results" = "Please change your filter.", check.names = FALSE))
    }
    
    datatable_employers %>%
      arrange(-n, organization,location_id) %>%
      left_join(records_locations,by="location_id", relationship="many-to-many") %>%
      select(organization, location_display_short, n) %>%
      rename(Count = n, 'Employer name' = organization, Location = location_display_short)
  })
  
  # Render the DataTable
  output$dataTablePub <- DT::renderDataTable({
    DT::datatable(dataTablePub(),
                  selection = list(mode = "single"),  # <-- only one row selectable
                  rownames = FALSE,
                  options = list(

                    orderClasses = TRUE,
                    pageLength = 25,
                    dom = 'tip',
                    initComplete = htmlwidgets::JS(
                      "function(settings, json) {",
                      "$(this.api().table().body()).css({'background-color': 'transparent'});",
                      "$(this.api().table().header()).css({'background-color': 'transparent'});",
                      "$('.dataTables_info').css({'background-color': 'transparent'});",
                      "$('.dataTables_paginate').css({'background-color': 'transparent'});",
                      "}"
                    )
                  ),escape=F
    )
  })
  
  # ------------------------------------------------------------------------------
  # Data Table: Colleges
  # ------------------------------------------------------------------------------
  
  dataTableCol <- reactive ({
    
    datatable_colleges <- filtered_colleges()
    
    # Check if the filtered data is empty
    if (is.null(datatable_colleges) || nrow(datatable_colleges) == 0) {
      return(data.frame("No results" = "Please change your filter.", check.names = FALSE))
    }
    
    datatable_colleges %>%
      arrange(-n, organization, location_id) %>%
      left_join(records_locations,by="location_id", relationship="many-to-many") %>%
      select(organization, location_display_short, n) %>%
      rename(Count = n, 'College name' = organization, Location = location_display_short)
  })
  
  # Render the DataTable
  output$dataTableCol <- DT::renderDataTable({
    req(has_started)
    DT::datatable(dataTableCol(),
                  selection = list(mode = "single"),  # <-- only one row selectable
                  rownames = FALSE,
                  options = list(

                    orderClasses = TRUE,
                    pageLength = 25,
                    dom = 'tip',
                    initComplete = htmlwidgets::JS(
                      "function(settings, json) {",
                      "$(this.api().table().body()).css({'background-color': 'transparent'});",
                      "$(this.api().table().header()).css({'background-color': 'transparent'});",
                      "$('.dataTables_info').css({'background-color': 'transparent'});",
                      "$('.dataTables_paginate').css({'background-color': 'transparent'});",
                      "}"
                    )
                  ),escape=F
    )
  })
  
  # ------------------------------------------------------------------------------
  # People Table UI
  # ------------------------------------------------------------------------------
  
  output$dynamic_ui_person <- renderUI({
    
    table_data <- dataTablePerson()
    
    # get clicked person row safely
    clicked_node <- table_data[input$dataTablePerson_rows_selected, ]
    location <- records_locations[records_locations$location_id==clicked_node$location_id,]
    # if (nrow(clicked_node) == 0) return(NULL)
    
    # aka_names <- filtered_persons() %>%
    #   filter(person_id %in% clicked_node$person_id & is.na(familyname_postal)==F & is.na(givenname_postal)==F) %>%
    #   # Combine postal names
    #   transmute(names_alt = paste(familyname_postal, givenname_postal)) %>%
    #   # Bind with existing names_alt column
    #   select(names_alt) %>%
    #   rbind(
    #     filtered_persons() %>%
    #       filter(person_id %in% clicked_node$person_id & is.na(names_alt) == F) %>%
    #       select(names_alt)
    #   ) %>%
    #   drop_na() %>%
    #   unique()
    
    # Degrees for clicked person
    
    if (!is.null(clicked_node) && nrow(clicked_node) > 0) {
      
      # --- Native place
      # native_loc <- records_locations %>%
      #   filter(location_id == clicked_node$location_id)
      
      # print(native_loc[,1]$location_name)
      
      clicked_person <- records_persons %>%
        filter(person_id == clicked_node$person_id) %>%
        slice(1)
      if (nrow(clicked_person) == 0) return(NULL)

      # --- Degree locations
      clicked_degrees <- records_education %>%
        filter(person_id == clicked_node$person_id) %>%
        filter(!if_all(-person_id, is.na)) %>%
        # Check if org actually resolves to a name
        left_join(records_organizations %>% select(organization_id, organization), by = "organization_id") %>%
        mutate(has_valid_org = !is.na(organization) & nzchar(organization)) %>%
        # Hide entries with no useful display info (no resolvable org AND no major)
        filter(has_valid_org | (!is.na(major_of_study) & nzchar(major_of_study) & major_of_study != "NULL")) %>%
        # Educational tier inferred from org name — used to order undated entries
        mutate(edu_level = case_when(
          grepl("小学|小學|尋常|Elementary School|Primary School", organization, ignore.case = TRUE) ~ 1L,
          grepl("中学|中學|中等|Middle School", organization, ignore.case = TRUE) &
            !grepl("大学|大學", organization, ignore.case = TRUE) ~ 2L,
          grepl("高等学校|高等學校|高校|高等女|師範|予科|予備|High School|Normal School|Preparatory", organization, ignore.case = TRUE) &
            !grepl("大学|大學", organization, ignore.case = TRUE) ~ 3L,
          grepl("大学|大學|専門|專門|College|University", organization, ignore.case = TRUE) ~ 4L,
          grepl("大学院|研究科|Graduate", organization, ignore.case = TRUE) ~ 5L,
          TRUE ~ 3L  # unknown defaults to middle tier
        )) %>%
        # Dated entries first (chronological); undated tail ordered by tier
        arrange(year_graduated, edu_level)
      
      # Jobs for clicked person
      person_volumes <- records_persons_core %>%
        filter(person_id == clicked_node$person_id) %>%
        pull(volume) %>% unlist()
      volume_year <- suppressWarnings(max(volume_year_lookup[person_volumes], na.rm = TRUE))
      if (is.infinite(volume_year)) volume_year <- NA_integer_
      clicked_jobs <- records_career %>%
        filter(person_id == clicked_node$person_id) %>%
        filter(!if_all(-person_id, is.na)) %>%
        mutate(sort_year = case_when(
          !is.na(start_year) ~ start_year,
          !is.na(current) & current == TRUE ~ volume_year,
          TRUE ~ NA_integer_
        )) %>%
        arrange(sort_year)
      
      clicked_family <- records_family %>%
        filter(person_id == clicked_node$person_id) %>%
        # Drop phantom rows where every display-relevant field is NA — earlier
        # guard kept them because gender/source_volume/source_page were filled,
        # producing empty <li></li> bullets at render time (e.g., P1935_425 R6/R14).
        filter(!(is.na(relation) & is.na(name) & is.na(name_latin) &
                 is.na(birth_year) & is.na(place) & is.na(location_id))) %>%
        arrange(order,order2,birth_year,order3,name,name_latin)
      
      # clicked_societies <- records_persons %>%
      #   filter(person_id %in% clicked_node$person_id) %>%
      #   select(societies,societies_date,societies_number,societies_status) %>% filter(!is.na(societies)) %>% unique() %>% arrange(societies)
      
      tagList(
        div(
          style = "display: flex; justify-content: space-between; align-items: flex-start;",
          h4(style = "margin-top: 0; margin-bottom: 0;",
             paste(
               
               if (!is.na(clicked_person$name_family_latin)) {
                 clicked_person$name_family_latin
               } else {
                 NULL
               },
               
               if (!is.na(clicked_person$name_given_latin)) {
                 clicked_person$name_given_latin
               } else {
                 NULL
               },
               
               # clicked_person$name_family_latin, clicked_person$name_given_latin,
               
               if (!is.na(clicked_person$name)) {
                 clicked_person$name
               } else {
                 NULL
               },
               
               if (!is.na(clicked_person$gender) && clicked_person$gender != "unknown") {
                 if (clicked_person$gender == "m") "♂"
                 else if (clicked_person$gender == "f") "♀"
                 else ""  # for unisex/unknown
               } else ""
               
               # if (!is.na(clicked_node$gender) && clicked_node$gender != "Unknown") {
               #   if (clicked_node$gender == "Male") "♂"
               #   else if (clicked_node$gender == "Female") "♀"
               #   else ""  # optional for unisex/unknown
               # } else ""
             )
          ),
          span(
            paste("ID #",clicked_person$person_id)
          )
        ),
        
        {
          # clicked_node$location_id is now the latest-residence after dataTablePerson
          # roll-up; use it for both the displayed text and the link target.
          loc_text <- if (!is.na(clicked_node$location_id)) format_location(location) else ""
          has_loc <- nzchar(loc_text)
          if (!is.na(clicked_person$birthyear) || has_loc) {
            p(HTML(paste0(
              if (!is.na(clicked_person$birthyear)) paste("born", clicked_person$birthyear) else NULL,
              if (has_loc)
                paste0(
                  if (!is.na(clicked_person$birthyear)) '<br>' else '',
                  'active in ',
                  '<a href="#" class="location-link" data-id="', clicked_node$location_id, '">',
                  loc_text,
                  '</a>'
                ) else NULL
            )))
          } else {
            NULL
          }
        },
        
        
        if (nrow(records_ranks %>% filter(person_id == clicked_person$person_id))>0) {
          clicked_ranks <- records_ranks %>% filter(person_id %in% clicked_person$person_id)
          tagList(
            p(HTML("<b>Rank</b>:"),
              HTML(paste(
                sapply(1:nrow(clicked_ranks), function(i) {
                  clicked_ranks$rank[i]}), collapse = ", ")))
          )
        } else "",

        if (!is.na(clicked_person$tax_amount) && nzchar(clicked_person$tax_amount)) {
          p(HTML(paste0("<b>Tax:</b> ", clicked_person$tax_amount, " yen",
                        if (!is.na(volume_year)) paste0(" (by ", volume_year, ")") else "")))
        } else "",

        if (!is.na(clicked_person$political_party) && nzchar(clicked_person$political_party)) {
          p(HTML(paste0("<b>Political affiliation:</b> ", clicked_person$political_party,
                        if (!is.na(volume_year)) paste0(" (by ", volume_year, ")") else "")))
        } else "",

        if (!is.na(clicked_person$religion)) {
          clicked_religion_rows <- records_religion %>%
            filter(person_id %in% clicked_person$person_id)
          religion_parts <- vapply(seq_len(nrow(clicked_religion_rows)), function(i) {
            r <- clicked_religion_rows$religion[i]
            sect <- religions_dict %>% filter(religion == r) %>% pull(sect)
            sect <- gsub(" \\(generic\\)", "", sect)
            label <- if (length(sect) == 0 || is.na(sect[1])) r else paste(sect[1], r)
            vols <- unlist(clicked_religion_rows$source_volume[i])
            yr <- suppressWarnings(max(volume_year_lookup[vols], na.rm = TRUE))
            if (is.finite(yr)) paste0(label, " (by ", yr, ")") else label
          }, character(1))
          p(HTML(paste0(
            "<b>Religious affiliation:</b> ",
            paste(religion_parts, collapse = ", ")
          )))
        } else "",

        # Only render if there is at least one degree
        if (nrow(clicked_degrees) > 0) {
          render_edu_item <- function(degree) {
            org_loc_id <- if (!is.na(degree$organization_id)) {
              records_organizations %>% filter(organization_id == degree$organization_id) %>% pull(location_id)
            } else character(0)
            org_loc_id <- if (length(org_loc_id) > 0 && !is.na(org_loc_id)) org_loc_id else NA
            location_row <- if (!is.na(org_loc_id)) {
              records_locations %>% filter(location_id == org_loc_id)
            } else data.frame()
            has_location <- nrow(location_row) > 0

            org_link <- ""
            has_org <- isTRUE(degree$has_valid_org)
            if (has_org) {
              org_link <- paste0(
                '<a href="#" class="college-link" data-id="', degree$organization_id, '">',
                degree$organization, '</a>'
              )
            }

            detail_parts <- c()
            if (!is.na(degree$major_of_study) && nzchar(degree$major_of_study) && degree$major_of_study != "NULL")
              detail_parts <- c(detail_parts, degree$major_of_study)
            if (!is.na(degree$year_graduated))
              detail_parts <- c(detail_parts, as.character(degree$year_graduated))
            if (has_location) {
              loc_text <- paste0(
                'in <a href="#" class="location-link" data-id="', org_loc_id, '">',
                format_location(location_row), '</a>'
              )
              detail_parts <- c(detail_parts, loc_text)
            }
            detail_text <- paste(detail_parts, collapse = ", ")

            paste0(
              "<li>",
              org_link,
              if (has_org && nzchar(detail_text)) paste0(" (", detail_text, ")")
              else if (!has_org && nzchar(detail_text)) detail_text
              else "",
              "</li>"
            )
          }

          tagList(
            p(HTML("<b>Education</b>:")),
            HTML(paste0(
              "<ul>",
              paste(sapply(1:nrow(clicked_degrees), function(i) render_edu_item(clicked_degrees[i, ])), collapse = ""),
              "</ul>"
            ))
          )
        } else {
          NULL
        },
        
        # Only render if there is at least one job
        if (nrow(clicked_jobs) > 0) {
          # Helper to render a single job item
          render_job_item <- function(job) {
            # Location fallback: career's own location_id, or org's location_id
            loc_id <- job$location_id
            if (is.na(loc_id) && !is.na(job$organization_id)) {
              org_loc <- records_organizations %>%
                filter(organization_id == job$organization_id) %>% pull(location_id)
              if (length(org_loc) > 0 && !is.na(org_loc)) loc_id <- org_loc
            }
            location_row <- if (!is.na(loc_id)) {
              records_locations %>% filter(location_id == loc_id)
            } else data.frame()
            has_location <- nrow(location_row) > 0

            # Build detail parts (year, current, location)
            is_current <- isTRUE(job$current)
            detail_parts <- c()
            if (!is.na(job$start_year)) {
              detail_parts <- c(detail_parts, as.character(job$start_year))
            } else if (is_current && !is.na(volume_year)) {
              detail_parts <- c(detail_parts, paste0("by ", volume_year))
            }
            if (is_current) detail_parts <- c(detail_parts, "current")
            if (has_location) {
              detail_parts <- c(detail_parts, paste0(
                'in <a href="#" class="location-link" data-id="', loc_id, '">',
                format_location(location_row), '</a>'
              ))
            }
            detail_text <- paste(detail_parts, collapse = ", ")

            paste0(
              "<li>",
              if (!is.na(job$job_title)) gsub("\\|", " & ", job$job_title) else "",
              if (!is.na(job$organization_id)) {
                organization <- records_organizations %>%
                  filter(organization_id == job$organization_id) %>%
                  pull(organization)
                if (length(organization) > 0) {
                  paste0(
                    ' at <a href="#" class="employer-link" data-id="', job$organization_id, '">',
                    organization, '</a>'
                  )
                } else ""
              } else "",
              if (nzchar(detail_text)) paste0(" (", detail_text, ")") else "",
              "</li>"
            )
          }

          # Split into dated/current (numbered) and undated (bulleted)
          jobs_dated <- clicked_jobs %>% filter(!is.na(sort_year))
          jobs_undated <- clicked_jobs %>% filter(is.na(sort_year))

          tagList(
            p(HTML("<b>Employments</b>:")),
            if (nrow(jobs_dated) > 0) {
              HTML(paste0(
                "<ol style='margin-bottom:0;'>",
                paste(sapply(1:nrow(jobs_dated), function(i) render_job_item(jobs_dated[i, ])), collapse = ""),
                "</ol>"
              ))
            } else NULL,
            if (nrow(jobs_undated) > 0) {
              HTML(paste0(
                "<ul style='margin-top:0;'>",
                paste(sapply(1:nrow(jobs_undated), function(i) render_job_item(jobs_undated[i, ])), collapse = ""),
                "</ul>"
              ))
            } else NULL
          )
        } else {
          NULL
        },
        
        # Family
        if (nrow(clicked_family) > 0) {
          render_family_item <- function(family_member) {
            pid <- family_member$person_id
            rid <- family_member$relation_id
            has_name <- !is.na(family_member$name_latin) || !is.na(family_member$name)

            # Look up education/career from separate tables
            fm_edu <- records_family_education %>%
              filter(person_id == pid, relation_id == rid) %>%
              left_join(records_organizations %>% select(organization_id, organization),
                        by = "organization_id")
            fm_car <- records_family_career %>%
              filter(person_id == pid, relation_id == rid) %>%
              left_join(records_organizations %>% select(organization_id, organization),
                        by = "organization_id")

            # Place: link if geolocated
            has_place <- !is.na(family_member$place) && nzchar(family_member$place)
            place_html <- ""
            if (has_place) {
              fm_loc_id <- if ("location_id" %in% names(family_member)) family_member$location_id else NA
              if (!is.na(fm_loc_id) && nzchar(fm_loc_id)) {
                loc_row <- records_locations %>% filter(location_id == fm_loc_id)
                if (nrow(loc_row) > 0) {
                  place_html <- paste0('<li>from <a href="#" class="location-link" data-id="',
                                       fm_loc_id, '">', format_location(loc_row), '</a></li>')
                } else {
                  place_html <- paste0("<li>from ", family_member$place, "</li>")
                }
              } else {
                place_html <- paste0("<li>from ", family_member$place, "</li>")
              }
            }

            # Education items
            edu_html <- ""
            if (nrow(fm_edu) > 0) {
              edu_html <- paste(sapply(1:nrow(fm_edu), function(i) {
                e <- fm_edu[i, ]
                has_org <- !is.na(e$organization_id) && !is.na(e$organization) && nzchar(e$organization)
                is_anaphoric_org <- has_org && grepl("^(Same|Within the same|Aforementioned|Above|Ibid)( |$)", e$organization)
                org_part <- if (is_anaphoric_org) {
                  paste0(' at ', e$organization)
                } else if (has_org) {
                  paste0(' at <a href="#" class="college-link" data-id="', e$organization_id, '">',
                         e$organization, '</a>')
                } else ""
                detail_parts <- c()
                if (!is.na(e$major_of_study) && nzchar(e$major_of_study) && e$major_of_study != "NULL")
                  detail_parts <- c(detail_parts, e$major_of_study)
                if (!is.na(e$year_graduated))
                  detail_parts <- c(detail_parts, as.character(e$year_graduated))
                prefix <- if (length(detail_parts) > 0 && any(grepl("\\D", detail_parts)))
                  "<li>graduated in " else "<li>studied"
                paste0(prefix, org_part,
                       if (length(detail_parts) > 0) paste0(" (", paste(detail_parts, collapse = ", "), ")")
                       else "", "</li>")
              }), collapse = "")
            }

            # Career items
            car_html <- ""
            if (nrow(fm_car) > 0) {
              car_html <- paste(sapply(1:nrow(fm_car), function(i) {
                j <- fm_car[i, ]
                has_org <- !is.na(j$organization_id) && !is.na(j$organization) && nzchar(j$organization)
                is_anaphoric_org <- has_org && grepl("^(Same|Within the same|Aforementioned|Above|Ibid)( |$)", j$organization)
                org_part <- if (is_anaphoric_org) {
                  paste0(' at ', j$organization)
                } else if (has_org) {
                  paste0(' at <a href="#" class="employer-link" data-id="', j$organization_id, '">',
                         j$organization, '</a>')
                } else ""
                job_part <- if (!is.na(j$job_title) && nzchar(j$job_title))
                  gsub("\\|", " & ", j$job_title) else ""
                detail_parts <- c()
                if (!is.na(j$start_year)) detail_parts <- c(detail_parts, as.character(j$start_year))
                paste0("<li>", if (nzchar(job_part)) paste0("worked as ", job_part) else "worked",
                       org_part,
                       if (length(detail_parts) > 0) paste0(" (", paste(detail_parts, collapse = ", "), ")")
                       else "", "</li>")
              }), collapse = "")
            }

            has_details <- nzchar(place_html) || nzchar(edu_html) || nzchar(car_html)

            paste0(
              "<li>",
              if (!is.na(family_member$name_latin)) paste0(gsub("\\|", " & ", family_member$name_latin)," ") else "",
              if (!is.na(family_member$name)) gsub("\\|", " & ", family_member$name) else "",
              if (!is.na(family_member$relation) & is.na(family_member$birth_year)) {
                rel_text <- gsub("\\|", " & ", family_member$relation)
                if (has_name) paste0(" (", rel_text, ")") else rel_text
              } else "",
              if (!is.na(family_member$relation) & !is.na(family_member$birth_year)) {
                rel_text <- gsub("\\|", " & ", family_member$relation)
                if (has_name) paste0(" (", rel_text, ", b. ", family_member$birth_year, ")")
                else paste0(rel_text, ", b. ", family_member$birth_year)
              } else "",
              if (is.na(family_member$relation) & !is.na(family_member$birth_year)) paste0(" (b. ",gsub("\\|", " & ", family_member$birth_year),")") else "",
              if (has_details) "<ul>" else "",
              place_html, edu_html, car_html,
              if (has_details) "</ul>" else "",
              "</li>"
            )
          }

          above_pattern <- "父|母|妻|夫|兄|姉|弟|妹|叔|伯|祖|岳|舅|姑|先|妃"
          family_above <- clicked_family %>%
            filter(!is.na(relation) & grepl(above_pattern, relation))

          # Forward/backward fill helpers for inferring birth_year on undated siblings
          .ffill <- function(x) {
            if (length(x) == 0) return(x); for (i in seq_along(x)[-1]) if (is.na(x[i])) x[i] <- x[i-1]; x
          }
          .bfill <- function(x) {
            n <- length(x); if (n < 2) return(x)
            for (i in seq.int(n - 1, 1, by = -1)) if (is.na(x[i])) x[i] <- x[i+1]; x
          }

          # Within each (order, sib_group), siblings of the same gender follow
          # 長/次/三/… rank; their unknown birth_years inherit a fractional value
          # from the neighbouring known year so the rank constraint isn't violated.
          # Across gender groups, the inferred birth_year alone decides order.
          family_below <- clicked_family %>%
            filter(is.na(relation) | !grepl(above_pattern, relation)) %>%
            mutate(sib_group = case_when(
              is.na(relation)         ~ "O",
              grepl("男", relation)   ~ "M",
              grepl("女|娘", relation) ~ "F",
              TRUE                     ~ "O"
            )) %>%
            group_by(order, sib_group) %>%
            arrange(order2, order3, .by_group = TRUE) %>%
            mutate(
              .yr_fwd = .ffill(as.numeric(birth_year)),
              .yr_bwd = .bfill(as.numeric(birth_year)),
              effective_year = case_when(
                !is.na(birth_year)               ~ as.numeric(birth_year),
                !is.na(.yr_fwd) & !is.na(.yr_bwd) ~ (.yr_fwd + .yr_bwd) / 2,
                !is.na(.yr_fwd)                   ~ .yr_fwd + 0.5,
                !is.na(.yr_bwd)                   ~ .yr_bwd - 0.5,
                TRUE                              ~ NA_real_
              )
            ) %>%
            ungroup() %>%
            select(-sib_group, -.yr_fwd, -.yr_bwd) %>%
            arrange(order, effective_year, order2, order3) %>%
            select(-effective_year)

          tagList(
            p(HTML("<b>Family</b>:")),
            if (nrow(family_above) > 0) {
              HTML(paste0(
                "<ul style='margin-bottom:0;'>",
                paste(sapply(1:nrow(family_above), function(i) render_family_item(family_above[i, ])), collapse = ""),
                "</ul>"
              ))
            } else NULL,
            if (nrow(family_below) > 0) {
              HTML(paste0(
                "<ul style='margin-top:0;'>",
                paste(sapply(1:nrow(family_below), function(i) render_family_item(family_below[i, ])), collapse = ""),
                "</ul>"
              ))
            } else NULL
          )
        } else {
          NULL
        },
        
        if (nrow(records_hobbies %>% filter(person_id == clicked_person$person_id))>0) {
          clicked_hobbies <- records_hobbies %>% filter(person_id %in% clicked_person$person_id) %>% arrange(hobby)
          tagList(
            p(HTML("<b>Hobbies</b>:"),
              HTML(paste(
                sapply(1:nrow(clicked_hobbies), function(i) {
                  clicked_hobbies$hobby[i]}), collapse = ", ")))
          )
        } else "",
        
        p(HTML(paste0(
          '<a href="#" class="entry-link" data-id="', clicked_node$person_id, '">',
          '🔍 Add as ID filter to existing query',
          '</a>'
        ))),

        plotOutput("person_map", height = "380px"),

        format_person_sources(clicked_person$person_id),
        
        # if (nrow(CERD_sources %>% 
        #          filter(person_id %in% clicked_node$person_id))>0) {
        #   p(
        #     HTML(
        #       paste0(
        #         "<span style='font-size:0.85em; color:#666;'>Source(s): ",
        #         paste((unique(CERD_sources %>% arrange(sources) %>%
        #                         filter(person_id %in% clicked_node$person_id))$sources),
        #               collapse = ", "),
        #         "</span>"
        #       )
        #     )
        #   )
        # } else {
        #   NULL
        # },
        
        # if (!is.na(clicked_node$mcbd_id)) {
        #   p(
        #     HTML(
        #       paste0(
        #         "<span style='font-size:0.85em; color:#666;'>This entry can also be found at <a href=\"https://heurist.huma-num.fr/ModernChinaBiographicalDatabase/\">MCBD</a> as ID #",
        #         clicked_node$mcbd_id,
        #         ".</span>"
        #       )
        #     )
        #   )
        # } else {
        #   NULL
        # }
        
      )
    } else {
      div(
        style = "text-align:center;",
        h4("Click on an entry for further information."))
    }
    
  })
  
  output$person_map <- renderPlot({
    # req(selected_node())  # you need a reactive for clicked_node
    
    # req(clicked_node())
    
    table_data <- dataTablePerson()
    clicked_node <- table_data[input$dataTablePerson_rows_selected, ]
    
    pid <- clicked_node$person_id
    locid <- clicked_node$location_id
    locid_edu <- records_education %>%
      filter(person_id %in% pid)
    locid_car <- records_career %>%
      filter(person_id %in% pid)
    # Fall back to the org's own location when the career row's location_id
    # is null but the linked organization has one — recovers ~30% of careers.
    if (nrow(locid_car) > 0) {
      org_loc_lookup <- records_organizations %>%
        filter(organization_id %in% locid_car$organization_id) %>%
        select(organization_id, location_id) %>%
        distinct(organization_id, .keep_all = TRUE) %>%
        deframe()
      locid_car <- locid_car %>%
        mutate(effective_location_id = coalesce(
          location_id,
          unname(org_loc_lookup[organization_id])
        ))
    } else {
      locid_car$effective_location_id <- character(0)
    }

    # Latest residence: prefer the most recent job's effective location, else
    # fall back to the person's recorded location_id (the previous "Native Place"
    # variable, which is unreliable as a true native place).
    person_volumes_pm <- records_persons_core %>%
      filter(person_id %in% pid) %>%
      pull(volume) %>% unlist()
    pm_volume_year <- suppressWarnings(max(volume_year_lookup[person_volumes_pm], na.rm = TRUE))
    if (is.infinite(pm_volume_year)) pm_volume_year <- NA_integer_

    latest_job_loc_id <- locid_car %>%
      filter(!is.na(effective_location_id)) %>%
      mutate(sort_year = case_when(
        current %in% TRUE  ~ as.integer(pm_volume_year),
        !is.na(start_year) ~ as.integer(start_year),
        TRUE               ~ NA_integer_
      )) %>%
      arrange(desc(sort_year)) %>%
      slice(1) %>%
      pull(effective_location_id)

    residence_loc_id <- if (length(latest_job_loc_id) > 0 && !is.na(latest_job_loc_id)) {
      latest_job_loc_id
    } else locid

    native_loc <- records_locations %>%
      filter(location_id %in% residence_loc_id) %>%
      mutate(type = "Latest residence")

    # Degree locations (via org → location)
    edu_loc_ids <- records_organizations %>%
      filter(organization_id %in% locid_edu$organization_id) %>%
      pull(location_id)
    degree_locs <- records_locations %>%
      filter(location_id %in% edu_loc_ids) %>%
      mutate(type = "College Location")

    # Job locations (career.location_id with org-level fallback)
    job_locs <- records_locations %>%
      filter(location_id %in% locid_car$effective_location_id) %>%
      mutate(type = "Job Location")

    # Combine all
    plot_data <- bind_rows(native_loc, degree_locs, job_locs) %>%
      filter(!st_is_empty(geometry) & !is.na(geometry)) %>%
      st_as_sf(sf_column_name = "geometry", crs = 4326)

    plot_data <- plot_data %>%
      filter(!location_name %in% c("日本", "中國", "台灣", "朝鮮"))

    plot_data <- plot_data %>%
      mutate(type = factor(type, levels = c("Latest residence", "College Location", "Job Location")))
    
    # Extract coordinates
    plot_data_coords <- plot_data %>%
      mutate(
        lon = st_coordinates(geometry)[,1],
        lat = st_coordinates(geometry)[,2]
      ) %>%
      st_drop_geometry()

    # Region detection by country code (more reliable than polygon intersect —
    # coastal points like Qingdao can fall just outside a simplified polygon).
    country_to_region <- c(ja = "japan", zh = "china", kr = "korea", tw = "taiwan")
    regions <- plot_data_coords %>%
      mutate(region = unname(country_to_region[country])) %>%
      filter(!is.na(region)) %>%
      pull(region) %>% unique()

    if (length(regions) == 0) return(NULL)
    
    maps <- list(
      china  = map_china,
      taiwan = map_taiwan,
      japan  = map_japan,
      korea  = map_korea
    )
    
    map_combined <- do.call(
      rbind,
      lapply(maps[regions], function(x) {
        st_sf(geometry = st_geometry(x))
      })
    )
    
    extents <- list(
      china  = list(x = c(103.5, 135.35), y = c(17.6, 53.6)),
      taiwan = list(x = c(119.8, 122.2), y = c(21.8, 25.4)),
      japan  = list(x = c(126.4, 146.2), y = c(29.5, 46)),
      korea  = list(x = c(124.0, 131.9), y = c(33.0, 43.0))
    )
    
    xlim <- range(unlist(lapply(extents[regions], `[[`, "x")))
    ylim <- range(unlist(lapply(extents[regions], `[[`, "y")))

    # Repel overlapping markers of different types: offset each type around the
    # shared centroid so a >50% overlap can't happen at the same coordinate.
    nudge_r <- 0.009 * diff(xlim)  # ~0.9% of viewport width
    plot_data_coords <- plot_data_coords %>%
      group_by(round(lon, 2), round(lat, 2)) %>%
      mutate(distinct_types = n_distinct(type)) %>%
      ungroup() %>%
      mutate(
        type_idx = as.integer(type),
        angle = (type_idx - 1) * 2 * pi / 3,
        lon = ifelse(distinct_types > 1, lon + nudge_r * cos(angle), lon),
        lat = ifelse(distinct_types > 1, lat + nudge_r * sin(angle), lat)
      ) %>%
      select(-distinct_types, -type_idx, -angle)

    ggplot() +
      geom_sf(
        data = map_combined,
        fill = "white",
        color = "grey50",
        linewidth = 0.2,
        linetype = "longdash"
      ) +
      geom_point(
        data = plot_data_coords,
        aes(x = lon, y = lat, color = type),
        size = 3,
        alpha = 0.7
      ) +
      scale_color_manual(values = c(
        "Latest residence"  = "#0072B2",
        "College Location"  = "#E69F00",
        "Job Location"      = "#009E73"
      )) +
      labs(color = "") +
      coord_sf(
        xlim   = xlim,
        ylim   = ylim,
        expand = FALSE
      ) +
      theme_minimal(base_size = 12) +
      theme(
        legend.position = "bottom",
        panel.background = element_blank(),
        plot.background  = element_blank(),
        panel.grid       = element_blank(),
        axis.title       = element_blank(),
        axis.text        = element_blank(),
        axis.ticks       = element_blank()
      )
    
  }, bg = "transparent")
  
  # ------------------------------------------------------------------------------
  # Employer Table UI
  # ------------------------------------------------------------------------------
  
  output$dynamic_ui_pubs <- renderUI({
    
    filtered_data <- filtered_employers()
    datatable_pubs <- filtered_employers() %>% arrange(-n, organization, location_id)
    
    clicked_node <- datatable_pubs[input$dataTablePub_rows_selected, ]
    location <- if (nrow(clicked_node) > 0 && !is.na(clicked_node$location_id)) {
      records_locations %>% filter(location_id == clicked_node$location_id)
    } else records_locations[0, ]

    if (!is.null(clicked_node) && nrow(clicked_node) > 0) {

      # Look up hierarchy info
      this_org <- records_organizations %>% filter(organization_id == clicked_node$organization_id)
      parent_org <- if (nrow(this_org) > 0 && !is.na(this_org$parent_organization_id[1])) {
        records_organizations %>% filter(organization_id == this_org$parent_organization_id[1])
      } else data.frame()
      child_orgs <- records_organizations %>%
        filter(!is.na(parent_organization_id) & parent_organization_id == clicked_node$organization_id)
      # Add career record counts per subdivision; drop children with no employees
      # (e.g., pure-education sub-units like 東大南校 / 東大歯科 — they share the
      # parent_organization_id but never appear in records_career, so they aren't
      # really sub-employers and would otherwise show up as misleading "0").
      if (nrow(child_orgs) > 0) {
        child_rec_counts <- filtered_jobs() %>%
          filter(organization_id %in% child_orgs$organization_id) %>%
          select(person_id, organization_id) %>%
          unique() %>%
          count(organization_id, name = "n_records")
        child_orgs <- child_orgs %>%
          inner_join(child_rec_counts, by = "organization_id") %>%
          arrange(desc(n_records), organization)
      }
      # HQ headcount: unique person_ids directly at this org (not subdivisions)
      n_hq <- filtered_jobs() %>%
        filter(organization_id == clicked_node$organization_id) %>%
        select(person_id) %>%
        distinct() %>%
        nrow()

      # Create dynamic UI
      tagList(

        div(
          style = "display: flex; justify-content: space-between; align-items: flex-start;",
          h4(style = "margin-top: 0; margin-bottom: 0;",
             HTML(paste0(clicked_node$organization))
          ),
          span(
            paste("ID #",clicked_node$organization_id)
          )
        ),

        if (!is.na(clicked_node$location_id) && nrow(location) > 0) {
          p(HTML(paste0(
            '<a href="#" class="location-link" data-id="', clicked_node$location_id, '">',
            format_location(location),
            '</a>'
          )))
        } else {
          NULL
        },

        # ISIC industry classification
        if (nrow(this_org) > 0 && !is.na(this_org$isic_label[1]) && nzchar(this_org$isic_label[1])) {
          p(HTML(paste0("<b>Industry:</b> ", this_org$isic_label[1],
                        " (ISIC ", this_org$isic_section[1], ")")))
        } else NULL,

        # Hierarchy: parent org
        if (nrow(parent_org) > 0) {
          p(HTML(paste0(
            'Part of <a href="#" class="employer-link" data-id="', parent_org$organization_id[1], '">',
            parent_org$organization[1],
            '</a>'
          )))
        } else NULL,

        # Employee headcount
        if (nrow(child_orgs) > 0 && n_hq > 0) {
          n_total <- n_hq + sum(child_orgs$n_records)
          p(HTML(paste0(
            '<b>Employees (HQ):</b> ', n_hq,
            ' &nbsp; <b>Total (incl. subdivisions):</b> ', n_total
          )))
        } else if (n_hq > 0) {
          p(HTML(paste0('<b>Employees:</b> ', n_hq)))
        } else NULL,

        # Hierarchy: child orgs (subdivisions)
        if (nrow(child_orgs) > 0) {
          n_children <- nrow(child_orgs)
          show_n <- min(n_children, 20)
          list_id <- paste0("subdiv_", gsub("[^a-zA-Z0-9]", "", clicked_node$organization_id))
          tagList(
            p(HTML("<b>Subdivisions:</b>")),
            HTML(paste0(
              "<ul>",
              paste(sapply(1:show_n, function(i) {
                paste0(
                  '<li><a href="#" class="employer-link" data-id="', child_orgs$organization_id[i], '">',
                  child_orgs$organization[i], '</a> (', child_orgs$n_records[i], ')</li>'
                )
              }), collapse = ""),
              if (n_children > show_n) paste0(
                '<div id="', list_id, '" style="display:none;">',
                paste(sapply((show_n + 1):n_children, function(i) {
                  paste0(
                    '<li><a href="#" class="employer-link" data-id="', child_orgs$organization_id[i], '">',
                    child_orgs$organization[i], '</a> (', child_orgs$n_records[i], ')</li>'
                  )
                }), collapse = ""),
                '</div>',
                '<li><a href="#" onclick="var el=document.getElementById(\'', list_id,
                '\');el.style.display=el.style.display===\'none\'?\'block\':\'none\';',
                'this.textContent=el.style.display===\'none\'?\'... and ', n_children - show_n,
                ' more\':\'(collapse)\';return false;">... and ',
                n_children - show_n, ' more</a></li>'
              ) else "",
              "</ul>"
            ))
          )
        } else NULL,

        p(HTML(paste0(
          '<a href="#" class="entry-link" data-id="', clicked_node$organization_id, '">',
          '🔍 Add as ID filter to existing query',
          '</a>'
        ))),

        plotOutput("employer_map", height = "380px")
      )
    } else {
      div(
        style = "text-align:center;",
        h4("Click on an entry for further information."))
    }
  })

  output$employer_map <- renderPlot({
    datatable_pubs <- filtered_employers() %>% arrange(-n, organization, location_id)
    clicked_node <- datatable_pubs[input$dataTablePub_rows_selected, ]
    req(clicked_node, nrow(clicked_node) > 0)

    org_id <- clicked_node$organization_id
    # Include subdivisions
    child_ids <- records_organizations %>%
      filter(!is.na(parent_organization_id) & parent_organization_id == org_id) %>%
      pull(organization_id)
    all_org_ids <- c(org_id, child_ids)

    # Distinct-employee count per org (HQ + each subdivision), respecting filters
    employees_per_org <- filtered_jobs() %>%
      filter(organization_id %in% all_org_ids) %>%
      select(person_id, organization_id) %>%
      distinct() %>%
      count(organization_id, name = "n")

    # Each org's own location_id (no event-fallback — only HQ / subdivision sites)
    org_locations <- records_organizations %>%
      filter(organization_id %in% all_org_ids) %>%
      select(organization_id, location_id) %>%
      distinct(organization_id, .keep_all = TRUE)

    loc_counts <- employees_per_org %>%
      left_join(org_locations, by = "organization_id") %>%
      filter(!is.na(location_id)) %>%
      group_by(location_id) %>%
      summarise(n = sum(n), .groups = "drop") %>%
      rename(effective_loc = location_id)

    plot_data <- records_locations %>%
      inner_join(loc_counts, by = c("location_id" = "effective_loc")) %>%
      filter(!st_is_empty(geometry) & !is.na(geometry)) %>%
      filter(!location_name %in% c("日本", "中國", "台灣", "朝鮮")) %>%
      st_as_sf(sf_column_name = "geometry", crs = 4326) %>%
      filter(lengths(st_intersects(geometry, map_asia)) > 0)

    # HQ pin: only the parent organisation's own location (subdivisions excluded)
    hq_sf <- if (!is.na(clicked_node$location_id)) {
      records_locations %>%
        filter(location_id == clicked_node$location_id) %>%
        filter(!st_is_empty(geometry) & !is.na(geometry)) %>%
        filter(!location_name %in% c("日本", "中國", "台灣", "朝鮮")) %>%
        st_as_sf(sf_column_name = "geometry", crs = 4326) %>%
        filter(lengths(st_intersects(geometry, map_asia)) > 0)
    } else NULL
    hq_df <- if (!is.null(hq_sf) && nrow(hq_sf) > 0) {
      hq_sf %>%
        mutate(lon = st_coordinates(geometry)[, 1],
               lat = st_coordinates(geometry)[, 2]) %>%
        st_drop_geometry() %>%
        select(lon, lat)
    } else data.frame(lon = numeric(), lat = numeric())

    if (nrow(plot_data) == 0 && nrow(hq_df) == 0) return(NULL)

    # Region detection across both employee dots and HQ pin so the zoom includes the HQ
    region_geom <- if (!is.null(hq_sf) && nrow(hq_sf) > 0) {
      bind_rows(plot_data %>% select(geometry), hq_sf %>% select(geometry)) %>%
        st_as_sf(sf_column_name = "geometry", crs = 4326)
    } else plot_data
    region_japan  <- region_geom %>% filter(lengths(st_intersects(geometry, map_japan)) > 0)
    region_taiwan <- region_geom %>% filter(lengths(st_intersects(geometry, map_taiwan)) > 0)
    region_china  <- region_geom %>% filter(lengths(st_intersects(geometry, map_china)) > 0)
    region_korea  <- region_geom %>% filter(lengths(st_intersects(geometry, map_korea)) > 0)

    regions <- c(
      china = nrow(region_china) > 0, taiwan = nrow(region_taiwan) > 0,
      japan = nrow(region_japan) > 0, korea = nrow(region_korea) > 0
    )
    regions <- names(regions)[regions]
    if (length(regions) == 0) return(NULL)
    # If japan is present and any non-Japan region is present, show all four
    if ("japan" %in% regions && length(setdiff(regions, "japan")) > 0) {
      regions <- c("china", "taiwan", "japan", "korea")
    }

    maps <- list(china = map_china, taiwan = map_taiwan, japan = map_japan, korea = map_korea)
    map_combined <- do.call(rbind, lapply(maps[regions], function(x) st_sf(geometry = st_geometry(x))))

    extents <- list(
      china  = list(x = c(103.5, 135.35), y = c(17.6, 53.6)),
      taiwan = list(x = c(119.8, 122.2),  y = c(21.8, 25.4)),
      japan  = list(x = c(126.4, 146.2),  y = c(29.5, 46)),
      korea  = list(x = c(124.0, 131.9),  y = c(33.0, 43.0))
    )
    xlim <- range(unlist(lapply(extents[regions], `[[`, "x")))
    ylim <- range(unlist(lapply(extents[regions], `[[`, "y")))

    plot_data_coords <- plot_data %>%
      mutate(lon = round(st_coordinates(geometry)[,1], 1),
             lat = round(st_coordinates(geometry)[,2], 1)) %>%
      st_drop_geometry() %>%
      group_by(lon, lat) %>%
      summarise(n = sum(n), .groups = "drop")

    p <- ggplot() +
      geom_sf(data = map_combined, fill = "white", color = "grey50", linewidth = 0.2, linetype = "longdash")
    if (nrow(plot_data_coords) > 0) {
      p <- p + geom_point(data = plot_data_coords,
                          aes(x = lon, y = lat, size = n, shape = "Employees"),
                          color = "#ff5964", alpha = 0.7)
    }
    if (nrow(hq_df) > 0) {
      p <- p + geom_point(data = hq_df,
                          aes(x = lon, y = lat, shape = "Headquarters"),
                          color = "black", fill = "#fff4a3",
                          size = 4.5, stroke = 1.1)
    }
    # Build the shape scale from layers actually present so override.aes lengths match
    shape_keys <- c(
      if (nrow(plot_data_coords) > 0) "Employees",
      if (nrow(hq_df)             > 0) "Headquarters"
    )
    shape_values <- c("Employees" = 16, "Headquarters" = 23)[shape_keys]
    shape_sizes  <- c("Employees" = 4,  "Headquarters" = 4.5)[shape_keys]
    shape_colors <- c("Employees" = "#ff5964", "Headquarters" = "black")[shape_keys]
    shape_fills  <- c("Employees" = NA, "Headquarters" = "#fff4a3")[shape_keys]
    p +
      scale_size_continuous(name = "n", range = c(0.6, 12),
                            breaks = c(1, 5, 10, 50, 100), trans = "sqrt") +
      scale_shape_manual(name = NULL,
                         values = shape_values,
                         guide = guide_legend(override.aes = list(
                           size = unname(shape_sizes),
                           color = unname(shape_colors),
                           fill = unname(shape_fills)
                         ))) +
      coord_sf(xlim = xlim, ylim = ylim, expand = FALSE) +
      theme_minimal(base_size = 12) +
      labs(title = "Employees at HQ and subdivisions", x = NULL, y = NULL) +
      theme(
        plot.title = element_text(face = "bold", size = 12),
        legend.position = "right",
        panel.background = element_blank(), plot.background = element_blank(),
        panel.grid = element_blank(), axis.title = element_blank(),
        axis.text = element_blank(), axis.ticks = element_blank()
      )
  }, bg = "transparent")

  # ------------------------------------------------------------------------------
  # College Table UI
  # ------------------------------------------------------------------------------
  
  output$dynamic_ui_cols <- renderUI({

    filtered_data <- filtered_colleges()
    datatable_cols <- filtered_colleges() %>% arrange(-n, organization, location_id)

    clicked_node <- datatable_cols[input$dataTableCol_rows_selected, ]
    location <- if (nrow(clicked_node) > 0 && !is.na(clicked_node$location_id)) {
      records_locations %>% filter(location_id == clicked_node$location_id)
    } else records_locations[0, ]

    if (!is.null(clicked_node) && nrow(clicked_node) > 0) {

      # Look up hierarchy info
      this_org <- records_organizations %>% filter(organization_id == clicked_node$organization_id)
      parent_org <- if (nrow(this_org) > 0 && !is.na(this_org$parent_organization_id[1])) {
        records_organizations %>% filter(organization_id == this_org$parent_organization_id[1])
      } else data.frame()
      child_orgs <- records_organizations %>%
        filter(!is.na(parent_organization_id) & parent_organization_id == clicked_node$organization_id)
      # Add student counts per subdivision; drop children with no students
      # (mirrors the employer panel — same parent in the data, but here we count
      # education records, so pure-employer subs are filtered out).
      if (nrow(child_orgs) > 0) {
        child_rec_counts <- filtered_degrees() %>%
          filter(organization_id %in% child_orgs$organization_id) %>%
          select(person_id, organization_id) %>%
          unique() %>%
          count(organization_id, name = "n_records")
        child_orgs <- child_orgs %>%
          inner_join(child_rec_counts, by = "organization_id") %>%
          arrange(desc(n_records), organization)
      }
      # HQ student count: unique person_ids enrolled directly at this org
      n_hq <- filtered_degrees() %>%
        filter(organization_id == clicked_node$organization_id) %>%
        select(person_id) %>%
        distinct() %>%
        nrow()

      tagList(

        div(
          style = "display: flex; justify-content: space-between; align-items: flex-start;",
          h4(style = "margin-top: 0; margin-bottom: 0;",
             HTML(paste0(clicked_node$organization))
          ),
          span(
            paste("ID #", clicked_node$organization_id)
          )
        ),

        if (!is.na(clicked_node$location_id) && nrow(location) > 0) {
          p(HTML(paste0(
            '<a href="#" class="location-link" data-id="', clicked_node$location_id, '">',
            format_location(location),
            '</a>'
          )))
        } else {
          NULL
        },

        # Hierarchy: parent org
        if (nrow(parent_org) > 0) {
          p(HTML(paste0(
            'Part of <a href="#" class="college-link" data-id="', parent_org$organization_id[1], '">',
            parent_org$organization[1],
            '</a>'
          )))
        } else NULL,

        # Student headcount
        if (nrow(child_orgs) > 0 && n_hq > 0) {
          n_total <- n_hq + sum(child_orgs$n_records)
          p(HTML(paste0(
            '<b>Students (HQ):</b> ', n_hq,
            ' &nbsp; <b>Total (incl. subdivisions):</b> ', n_total
          )))
        } else if (n_hq > 0) {
          p(HTML(paste0('<b>Students:</b> ', n_hq)))
        } else NULL,

        # Hierarchy: child orgs (subdivisions)
        if (nrow(child_orgs) > 0) {
          n_children <- nrow(child_orgs)
          show_n <- min(n_children, 20)
          list_id <- paste0("subdiv_col_", gsub("[^a-zA-Z0-9]", "", clicked_node$organization_id))
          tagList(
            p(HTML("<b>Subdivisions:</b>")),
            HTML(paste0(
              "<ul>",
              paste(sapply(1:show_n, function(i) {
                paste0(
                  '<li><a href="#" class="college-link" data-id="', child_orgs$organization_id[i], '">',
                  child_orgs$organization[i], '</a> (', child_orgs$n_records[i], ')</li>'
                )
              }), collapse = ""),
              if (n_children > show_n) paste0(
                '<div id="', list_id, '" style="display:none;">',
                paste(sapply((show_n + 1):n_children, function(i) {
                  paste0(
                    '<li><a href="#" class="college-link" data-id="', child_orgs$organization_id[i], '">',
                    child_orgs$organization[i], '</a> (', child_orgs$n_records[i], ')</li>'
                  )
                }), collapse = ""),
                '</div>',
                '<li><a href="#" onclick="var el=document.getElementById(\'', list_id,
                '\');el.style.display=el.style.display===\'none\'?\'block\':\'none\';',
                'this.textContent=el.style.display===\'none\'?\'... and ', n_children - show_n,
                ' more\':\'(collapse)\';return false;">... and ',
                n_children - show_n, ' more</a></li>'
              ) else "",
              "</ul>"
            ))
          )
        } else NULL,

        p(HTML(paste0(
          '<a href="#" class="entry-link" data-id="', clicked_node$organization_id, '">',
          '🔍 Add as ID filter to existing query',
          '</a>'
        ))),

        plotOutput("college_map", height = "380px")

      )
    } else {
      div(
        style = "text-align:center;",
        h4("Click on an entry for further information."))
    }
  })
  
  # # --- Then define the radar plot separately in server
  # output$college_radar <- renderPlot({
  #   
  #   # 5 main disciplines
  #   disciplines <- c("CE 🏗️", "ME 🔧", "EE 💡️️", "MiE ⛏️", "ChE 🧪")
  #   # Prepare counts per discipline for selected college
  #   radar_counts <- filtered_degrees() %>%
  #     filter(organization_id == selected_node_col()$organization_id) %>%
  #     mutate(discipline = case_when(
  #       grepl("Civil", field, ignore.case = TRUE) ~ "CE 🏗️",
  #       grepl("Mechanical", field, ignore.case = TRUE) ~ "ME 🔧",
  #       grepl("Electrical", field, ignore.case = TRUE) ~ "EE 💡️️",
  #       grepl("Mining", field, ignore.case = TRUE) ~ "MiE ⛏️",
  #       grepl("Chemical", field, ignore.case = TRUE) ~ "ChE 🧪",
  #       TRUE ~ NA_character_
  #     )) %>%
  #     drop_na(discipline) %>%
  #     count(discipline) %>%
  #     complete(discipline = disciplines, fill = list(n = 0)) %>%
  #     arrange(match(discipline, disciplines)) 
  #   
  #   # Create dataframe for fmsb radarchart
  #   radar_data <- data.frame(
  #     rbind(
  #       rep(max(radar_counts$n, 1), length(disciplines)),  # max row
  #       rep(0, length(disciplines)),                       # min row
  #       radar_counts$n                                     # actual counts
  #     )
  #   )
  #   colnames(radar_data) <- disciplines
  #   
  #   # Draw the radar chart
  #   par(mar = c(0, 0, 0, 0))  # remove all inner margins
  #   radarchart(
  #     radar_data,
  #     cglcol = "grey", cglty = 1, cglwd = 0.8
  #   )
  # },
  # bg = "transparent")
  
  # ------------------------------------------------------------------------------
  # College Map (mirrors employer_map but uses education data)
  # ------------------------------------------------------------------------------
  output$college_map <- renderPlot({
    datatable_cols <- filtered_colleges() %>% arrange(-n, organization, location_id)
    clicked_node <- datatable_cols[input$dataTableCol_rows_selected, ]
    req(clicked_node, nrow(clicked_node) > 0)

    org_id <- clicked_node$organization_id

    # Collect locations using each student's latest residence:
    # primary = latest job location, fallback 1 = native location_id,
    # fallback 2 = the school's own location_id.
    edu_locs <- records_education %>%
      filter(organization_id == org_id)

    if (nrow(edu_locs) > 0) {
      org_loc_id <- records_organizations %>%
        filter(organization_id == org_id) %>%
        pull(location_id)
      org_loc_id <- if (length(org_loc_id) > 0) org_loc_id[1] else NA_character_

      student_pids <- unique(edu_locs$person_id)

      careers_for_students <- records_career %>%
        filter(person_id %in% student_pids)
      if (nrow(careers_for_students) > 0) {
        org_loc_lookup_s <- records_organizations %>%
          filter(organization_id %in% careers_for_students$organization_id) %>%
          select(organization_id, location_id) %>%
          distinct(organization_id, .keep_all = TRUE) %>%
          deframe()
        careers_for_students <- careers_for_students %>%
          mutate(effective_location_id = coalesce(
            location_id,
            unname(org_loc_lookup_s[organization_id])
          ))
      } else {
        careers_for_students$effective_location_id <- character(0)
      }
      latest_job_per_student <- careers_for_students %>%
        filter(!is.na(effective_location_id)) %>%
        group_by(person_id) %>%
        arrange(
          desc(!is.na(start_year)),
          desc(start_year),
          desc(current %in% TRUE),
          .by_group = TRUE
        ) %>%
        slice(1) %>%
        ungroup() %>%
        select(person_id, latest_loc = effective_location_id)

      residence_per_student <- records_persons_core %>%
        filter(person_id %in% student_pids) %>%
        select(person_id, native_loc = location_id) %>%
        distinct() %>%
        left_join(latest_job_per_student, by = "person_id") %>%
        mutate(residence_loc = coalesce(latest_loc, native_loc)) %>%
        select(person_id, residence_loc)

      loc_counts <- edu_locs %>%
        left_join(residence_per_student, by = "person_id") %>%
        mutate(effective_loc = coalesce(residence_loc, org_loc_id)) %>%
        filter(!is.na(effective_loc)) %>%
        count(effective_loc, name = "n")
    } else {
      loc_counts <- records_organizations %>%
        filter(organization_id == org_id) %>%
        filter(!is.na(location_id)) %>%
        select(effective_loc = location_id) %>%
        unique() %>%
        mutate(n = 1L)
    }

    if (nrow(loc_counts) == 0) return(NULL)

    plot_data <- records_locations %>%
      inner_join(loc_counts, by = c("location_id" = "effective_loc")) %>%
      filter(!st_is_empty(geometry) & !is.na(geometry)) %>%
      filter(!location_name %in% c("日本", "中國", "台灣", "朝鮮")) %>%
      st_as_sf(sf_column_name = "geometry", crs = 4326) %>%
      filter(lengths(st_intersects(geometry, map_asia)) > 0)

    # Campus pin: only the school's own location (no subdivisions)
    hq_sf <- if (!is.na(clicked_node$location_id)) {
      records_locations %>%
        filter(location_id == clicked_node$location_id) %>%
        filter(!st_is_empty(geometry) & !is.na(geometry)) %>%
        filter(!location_name %in% c("日本", "中國", "台灣", "朝鮮")) %>%
        st_as_sf(sf_column_name = "geometry", crs = 4326) %>%
        filter(lengths(st_intersects(geometry, map_asia)) > 0)
    } else NULL
    hq_df <- if (!is.null(hq_sf) && nrow(hq_sf) > 0) {
      hq_sf %>%
        mutate(lon = st_coordinates(geometry)[, 1],
               lat = st_coordinates(geometry)[, 2]) %>%
        st_drop_geometry() %>%
        select(lon, lat)
    } else data.frame(lon = numeric(), lat = numeric())

    if (nrow(plot_data) == 0 && nrow(hq_df) == 0) return(NULL)

    region_geom <- if (!is.null(hq_sf) && nrow(hq_sf) > 0) {
      bind_rows(plot_data %>% select(geometry), hq_sf %>% select(geometry)) %>%
        st_as_sf(sf_column_name = "geometry", crs = 4326)
    } else plot_data
    region_japan  <- region_geom %>% filter(lengths(st_intersects(geometry, map_japan)) > 0)
    region_taiwan <- region_geom %>% filter(lengths(st_intersects(geometry, map_taiwan)) > 0)
    region_china  <- region_geom %>% filter(lengths(st_intersects(geometry, map_china)) > 0)
    region_korea  <- region_geom %>% filter(lengths(st_intersects(geometry, map_korea)) > 0)

    regions <- c(
      china = nrow(region_china) > 0, taiwan = nrow(region_taiwan) > 0,
      japan = nrow(region_japan) > 0, korea = nrow(region_korea) > 0
    )
    regions <- names(regions)[regions]
    if (length(regions) == 0) return(NULL)
    if ("japan" %in% regions && length(setdiff(regions, "japan")) > 0) {
      regions <- c("china", "taiwan", "japan", "korea")
    }

    maps <- list(china = map_china, taiwan = map_taiwan, japan = map_japan, korea = map_korea)
    map_combined <- do.call(rbind, lapply(maps[regions], function(x) st_sf(geometry = st_geometry(x))))

    extents <- list(
      china  = list(x = c(103.5, 135.35), y = c(17.6, 53.6)),
      taiwan = list(x = c(119.8, 122.2),  y = c(21.8, 25.4)),
      japan  = list(x = c(126.4, 146.2),  y = c(29.5, 46)),
      korea  = list(x = c(124.0, 131.9),  y = c(33.0, 43.0))
    )
    xlim <- range(unlist(lapply(extents[regions], `[[`, "x")))
    ylim <- range(unlist(lapply(extents[regions], `[[`, "y")))

    plot_data_coords <- plot_data %>%
      mutate(lon = round(st_coordinates(geometry)[,1], 1),
             lat = round(st_coordinates(geometry)[,2], 1)) %>%
      st_drop_geometry() %>%
      group_by(lon, lat) %>%
      summarise(n = sum(n), .groups = "drop")

    p <- ggplot() +
      geom_sf(data = map_combined, fill = "white", color = "grey50", linewidth = 0.2, linetype = "longdash")
    if (nrow(plot_data_coords) > 0) {
      p <- p + geom_point(data = plot_data_coords,
                          aes(x = lon, y = lat, size = n, shape = "Students"),
                          color = "#4a90d9", alpha = 0.7)
    }
    if (nrow(hq_df) > 0) {
      p <- p + geom_point(data = hq_df,
                          aes(x = lon, y = lat, shape = "Campus"),
                          color = "black", fill = "#fff4a3",
                          size = 4.5, stroke = 1.1)
    }
    shape_keys <- c(
      if (nrow(plot_data_coords) > 0) "Students",
      if (nrow(hq_df)             > 0) "Campus"
    )
    shape_values <- c("Students" = 16, "Campus" = 23)[shape_keys]
    shape_sizes  <- c("Students" = 4,  "Campus" = 4.5)[shape_keys]
    shape_colors <- c("Students" = "#4a90d9", "Campus" = "black")[shape_keys]
    shape_fills  <- c("Students" = NA, "Campus" = "#fff4a3")[shape_keys]
    p +
      scale_size_continuous(name = "n", range = c(0.6, 12),
                            breaks = c(1, 5, 10, 50, 100), trans = "sqrt") +
      scale_shape_manual(name = NULL,
                         values = shape_values,
                         guide = guide_legend(override.aes = list(
                           size = unname(shape_sizes),
                           color = unname(shape_colors),
                           fill = unname(shape_fills)
                         ))) +
      coord_sf(xlim = xlim, ylim = ylim, expand = FALSE) +
      theme_minimal(base_size = 12) +
      labs(title = "Latest student activity", x = NULL, y = NULL) +
      theme(
        plot.title = element_text(face = "bold", size = 12),
        legend.position = "right",
        panel.background = element_blank(), plot.background = element_blank(),
        panel.grid = element_blank(), axis.title = element_blank(),
        axis.text = element_blank(), axis.ticks = element_blank()
      )
  }, bg = "transparent")

  # ------------------------------------------------------------------------------
  # Visualization: Fields Distribution Bar Chart
  # ------------------------------------------------------------------------------
  output$barChart_fields <- renderPlot({
    
    religion_colors <- c(
      "Buddhism"      = "#E69F00",
      "Christianity" = "#0072B2",
      "Confucianism" = "#8C6D31",
      "Shintō"       = "#C73A2C",
      "Tenrikyō"     = "#009E73"
    )
    
    # Map faculty codes to German display labels and filter by publication authors
    
    filtered_persons() %>%
      select(religion, person_id) %>%
      unique() %>%
      left_join(religions_dict, by = "religion") %>%
      drop_na(belongs_to) %>%
      count(school, belongs_to, sort = TRUE) %>%
      slice_head(n = 10) %>%
      ggplot(aes(x = reorder(school, n), y = n, fill = belongs_to)) +
      geom_col() +
      geom_text(
        aes(label = paste0(" ",school)),
        y = 0,
        hjust = 0,
        size = 6,
        color = "black"
      )+
      coord_flip() +
      scale_y_continuous(labels = scales::label_comma()) +
      labs(
        x = NULL,
        y = "Count",
        fill = "Religion",
        title = "Religious affiliations of queried individuals",
        caption = "Query via: GoA-DB"
      ) +
      scale_fill_manual(values = religion_colors) +
      theme_minimal() +
      theme(
        axis.text.y  = element_blank(),
        axis.ticks.y = element_blank(),
        plot.title = element_text(face = "bold", size = 14),
        legend.text = element_text(size = 10)
      )
    
  })
  
  # ------------------------------------------------------------------------------
  # Shared reactive computations
  # ------------------------------------------------------------------------------
  
  # Base table: each person mapped to the province of their LATEST residence —
  # primary = most recent job's location_id (start_year max, current=true tiebreaker);
  # fallback = the person's own location_id (the historically misnamed "native" field).
  stat_region_base <- reactive({
    req(filtered_persons(), filtered_locations())

    pids <- filtered_persons()$person_id

    careers_for_pids <- records_career %>%
      filter(person_id %in% pids)
    if (nrow(careers_for_pids) > 0) {
      org_loc_lookup <- records_organizations %>%
        filter(organization_id %in% careers_for_pids$organization_id) %>%
        select(organization_id, location_id) %>%
        distinct(organization_id, .keep_all = TRUE) %>%
        deframe()
      careers_for_pids <- careers_for_pids %>%
        mutate(effective_location_id = coalesce(
          location_id,
          unname(org_loc_lookup[organization_id])
        ))
    } else {
      careers_for_pids$effective_location_id <- character(0)
    }

    latest_job <- careers_for_pids %>%
      filter(!is.na(effective_location_id)) %>%
      group_by(person_id) %>%
      arrange(
        desc(!is.na(start_year)),
        desc(start_year),
        desc(current %in% TRUE),
        .by_group = TRUE
      ) %>%
      slice(1) %>%
      ungroup() %>%
      select(person_id, latest_loc = effective_location_id)

    loc_to_province <- records_locations %>%
      st_drop_geometry() %>%
      select(location_id, province)

    filtered_persons() %>%
      select(person_id, native_loc = location_id) %>%
      distinct() %>%
      left_join(latest_job, by = "person_id") %>%
      mutate(residence_loc = coalesce(latest_loc, native_loc)) %>%
      left_join(loc_to_province, by = c("residence_loc" = "location_id")) %>%
      select(province, person_id) %>%
      distinct()
  })
  
  # Province-level counts (used for Asia map)
  province_counts <- reactive({
    stat_region_base() %>%
      filter(!is.na(province)) %>%
      count(province, name = "n") %>%
      arrange(desc(n))
  })
  
  # # Country-level counts (used for World map)
  # country_counts <- reactive({
  #   stat_region_base() %>%
  #     filter(!is.na(country)) %>%
  #     count(country, name = "n") %>%
  #     arrange(desc(n))
  # })
  
  # Colleges
  stat_region_college <- reactive({
    req(filtered_degrees(), filtered_colleges(), records_locations)
    
    filtered_degrees() %>%
      select(organization_id, person_id) %>%
      left_join(filtered_colleges() %>% select(organization_id, location_id),
                by = "organization_id", relationship = "many-to-many") %>%
      left_join(records_locations %>%
                  
                  filter(!location_name %in% c("日本", "中國", "台灣", "朝鮮")) %>%
                  
                  select(location_id, geometry),
                by = "location_id", relationship = "many-to-many") %>%
      select(person_id, geometry) %>%
      count(geometry, name = "n") %>%
      drop_na() %>%
      mutate(type = "College")
  })
  
  # Employers
  stat_region_employer <- reactive({
    req(filtered_jobs(), filtered_employers(), records_locations)
    
    filtered_jobs() %>%
      select(organization_id, person_id) %>%
      left_join(filtered_employers() %>% select(organization_id, location_id),
                by = "organization_id", relationship = "many-to-many") %>%
      left_join(records_locations  %>%
                  
                  filter(!location_name %in% c("日本", "中國", "台灣", "朝鮮")) %>%
                  
                  select(location_id, geometry, location_name)
                  
                  ,
                by = "location_id", relationship = "many-to-many") %>%
      select(person_id, geometry) %>%
      count(geometry, name = "n") %>%
      drop_na() %>%
      mutate(type = "Employer")
  })
  
  # ------------------------------------------------------------------------------
  # Visualization: Asia Map
  # ------------------------------------------------------------------------------
  
  # output$barChart_region <- renderPlot({
  #   req(province_counts())
  # 
  #   stat_region <- map_asia %>%
  #     left_join(province_counts(), by = "province")
  #   
  #   ggplot() +
  #     geom_sf(data = map_asia, fill = "gray95", color = "gray95", size = 0.1) +
  #     geom_sf(data = stat_region, aes(fill = n), color = "white", size = 0.2) +
  #     geom_sf(data = stat_region_employer(),
  #             aes(geometry = geometry, size = n, color = type),
  #             alpha = 0.9, shape = 1) +
  #     geom_sf(data = stat_region_college(),
  #             aes(geometry = geometry, size = n, color = type),
  #             alpha = 0.9, shape = 4) +
  #     coord_sf(xlim = c(73, 149), ylim = c(16.5, 54.5)) +
  #     scale_fill_gradient(low = "#FFD7DA", high = "#ff5964",
  #                         na.value = "grey90", name = "Native province") +
  #     scale_color_manual(values = c("College" = "darkgray", "Employer" = "darkgray"),
  #                        name = "Event type") +
  #     scale_size_continuous(name = "Events per location",
  #                           breaks = c(1, 10, 100, 250, 500),
  #                           range = c(2, 10)) +
  #     theme_minimal() +
  #     labs(
  #       title = "Geographic mobility of queried individuals",
  #       caption = "Query via: GoA-DB",
  #       x = NULL, y = NULL
  #     ) +
  #     theme(
  #       legend.position = "right",
  #       plot.title = element_text(face = "bold", size = 14),
  #       legend.text = element_text(size = 10)
  #     )
  # })
  
  # ------------------------------------------------------------------------------
  # Visualization: Japan Map
  # ------------------------------------------------------------------------------
  
  japan_zoom_coords <- list(
    "All Japan"        = list(xlim = c(126.5, 146.8), ylim = c(24.25, 47.75)),
    "Chubu"            = list(xlim = c(136.0, 139.5), ylim = c(34.5, 38.5)),
    "Chugoku/Shikoku"  = list(xlim = c(130.5, 134.5), ylim = c(32.0, 36.5)),
    "Hokkaido"         = list(xlim = c(139.5, 146.0), ylim = c(41.0, 48.5)),
    "Kansai"           = list(xlim = c(134.0, 136.5), ylim = c(33.5, 36.5)),
    "Kanto"            = list(xlim = c(138.7, 141.0), ylim = c(34.5, 37.2)),
    "Kyushu"           = list(xlim = c(129.0, 132.5), ylim = c(30.5, 34.5)),
    "Okinawa"          = list(xlim = c(126.0, 129.0), ylim = c(24.0, 27.5)),
    "Tohoku"           = list(xlim = c(139.0, 142.0), ylim = c(37.0, 41.0))
  )

  japan_zoom_cities <- list(
    "Chubu"           = data.frame(city = "Nagoya",    lon = 136.91, lat = 35.18),
    "Chugoku/Shikoku" = data.frame(city = "Hiroshima", lon = 132.46, lat = 34.40),
    "Hokkaido"        = data.frame(city = "Sapporo",   lon = 141.35, lat = 43.06),
    "Kansai"          = data.frame(city = "Osaka",     lon = 135.50, lat = 34.69),
    "Kanto"           = data.frame(city = "Tokyo",     lon = 139.69, lat = 35.69),
    "Kyushu"          = data.frame(city = "Fukuoka",   lon = 130.40, lat = 33.59),
    "Okinawa"         = data.frame(city = "Naha",      lon = 127.68, lat = 26.33),
    "Tohoku"          = data.frame(city = "Sendai",    lon = 140.87, lat = 38.27)
  )

  asia_zoom_coords <- list(
    "All East Asia"  = list(xlim = c(103, 135), ylim = c(17.5, 54)),
    "East China"     = list(xlim = c(112, 124), ylim = c(25, 42)),
    "Korea"          = list(xlim = c(124, 131), ylim = c(33, 41)),
    "Manchuria"      = list(xlim = c(120, 134), ylim = c(38, 54)),
    "South China"    = list(xlim = c(106, 119), ylim = c(17, 32)),
    "Taiwan"         = list(xlim = c(119, 123), ylim = c(21.5, 26))
  )

  asia_zoom_cities <- list(
    "East China"  = data.frame(city = "Beijing",  lon = 116.40, lat = 39.90),
    "Korea"       = data.frame(city = "Seoul",     lon = 126.98, lat = 37.57),
    "Manchuria"   = data.frame(city = "Harbin",    lon = 126.65, lat = 45.75),
    "South China" = data.frame(city = "Guangzhou",  lon = 113.26, lat = 23.13),
    "Taiwan"      = data.frame(city = "Taipei",    lon = 121.57, lat = 25.04)
  )

  output$barChart_region_japan <- renderPlot({
    req(province_counts())

    japan_zoom_sel <- input$japan_zoom %||% "All Japan"
    bounds <- japan_zoom_coords[[japan_zoom_sel]]

    stat_region <- map_japan %>%
      left_join(province_counts(), by = "province")

    p <- ggplot() +
      geom_sf(data = map_japan, fill = "gray95", color = "gray95", size = 0.1) +
      geom_sf(data = stat_region, aes(fill = n), color = "white", size = 0.2) +
      geom_sf(data = stat_region_employer() %>% filter(lengths(st_intersects(geometry, map_japan)) > 0),
              aes(geometry = geometry, size = n, color = type),
              alpha = 0.55, shape = 1) +
      geom_sf(data = stat_region_college() %>% filter(lengths(st_intersects(geometry, map_japan)) > 0),
              aes(geometry = geometry, size = n, color = type),
              alpha = 0.55, shape = 4)

    if (japan_zoom_sel != "All Japan" && japan_zoom_sel %in% names(japan_zoom_cities)) {
      p <- p + geom_text(data = japan_zoom_cities[[japan_zoom_sel]],
                         aes(x = lon, y = lat, label = city),
                         size = 3.5, fontface = "bold", color = "#333333",
                         nudge_y = 0.08, check_overlap = TRUE)
    }

    p + coord_sf(xlim = bounds$xlim, ylim = bounds$ylim) +
      scale_fill_gradient(low = "#FFD7DA", high = "#ff5964",
                          trans = "log10",
                          labels = function(x) format(x, big.mark = ",", scientific = FALSE, trim = TRUE),
                          na.value = "grey90", name = "Latest residence") +
      scale_color_manual(values = c("College" = "#404040", "Employer" = "#404040"),
                         name = "Event type") +
      scale_size_continuous(name = "Events per location",
                            breaks = c(1, 10, 100, 250, 500),
                            range = c(1.5, 14),
                            trans = "sqrt") +
      theme_minimal() +
      labs(
        title = "Geographic mobility of queried individuals (Japan)",
        caption = "Query via: GoA-DB",
        x = NULL, y = NULL
      ) +
      theme(
        legend.position = "right",
        plot.title = element_text(face = "bold", size = 14),
        legend.text = element_text(size = 10)
      )
  })
  
  # ------------------------------------------------------------------------------
  # Visualization: Asia Map
  # ------------------------------------------------------------------------------
  
  output$barChart_region_asia <- renderPlot({
    req(province_counts())

    asia_zoom_sel <- input$asia_zoom %||% "All East Asia"
    bounds <- asia_zoom_coords[[asia_zoom_sel]]

    stat_region <- map_asia_outer %>%
      left_join(province_counts(), by = "province")

    p <- ggplot() +
      geom_sf(data = map_asia_outer, fill = "gray95", color = "gray95", size = 0.1) +
      geom_sf(data = stat_region, aes(fill = n), color = "white", size = 0.2) +
      geom_sf(data = stat_region_employer() %>% filter(lengths(st_intersects(geometry, map_japan)) < 1 & lengths(st_intersects(geometry, map_asia)) > 0),
              aes(geometry = geometry, size = n, color = type),
              alpha = 0.55, shape = 1) +
      geom_sf(data = stat_region_college() %>% filter(lengths(st_intersects(geometry, map_japan)) < 1 & lengths(st_intersects(geometry, map_asia)) > 0),
              aes(geometry = geometry, size = n, color = type),
              alpha = 0.55, shape = 4)

    if (asia_zoom_sel != "All East Asia" && asia_zoom_sel %in% names(asia_zoom_cities)) {
      p <- p + geom_text(data = asia_zoom_cities[[asia_zoom_sel]],
                         aes(x = lon, y = lat, label = city),
                         size = 3.5, fontface = "bold", color = "#333333",
                         nudge_y = 0.08, check_overlap = TRUE)
    }

    p + coord_sf(xlim = bounds$xlim, ylim = bounds$ylim) +
      scale_fill_gradient(low = "#FFD7DA", high = "#ff5964",
                          trans = "log10",
                          labels = function(x) format(x, big.mark = ",", scientific = FALSE, trim = TRUE),
                          na.value = "grey90", name = "Latest residence") +
      scale_color_manual(values = c("College" = "#404040", "Employer" = "#404040"),
                         name = "Event type") +
      scale_size_continuous(name = "Events per location",
                            breaks = c(1, 10, 100, 250, 500),
                            range = c(1.5, 14),
                            trans = "sqrt") +
      theme_minimal() +
      labs(
        title = "Geographic mobility of queried individuals (East Asia)",
        caption = "Query via: GoA-DB",
        x = NULL, y = NULL
      ) +
      theme(
        legend.position = "right",
        plot.title = element_text(face = "bold", size = 14),
        legend.text = element_text(size = 10)
      )
  })
  
  # ------------------------------------------------------------------------------
  # Visualization: World Map
  # ------------------------------------------------------------------------------
  
  # output$barChart_region_world <- renderPlot({
  #   req(country_counts())
  #   
  #   stat_region <- world_1938 %>%
  #     rename(country = NAME) %>%
  #     mutate(country = trimws(as.character(country))) %>%
  #     left_join(country_counts() %>%
  #                 mutate(country = trimws(as.character(country))),
  #               by = "country")
  #   
  #   ggplot(stat_region) +
  #     geom_sf(aes(fill = n), color = "white", size = 0.2) +
  #     geom_sf(data = stat_region_employer(),
  #             aes(geometry = geometry, size = n, color = type),
  #             alpha = 0.6) +
  #     geom_sf(data = stat_region_college(),
  #             aes(geometry = geometry, size = n, color = type),
  #             alpha = 0.4) +
  #     coord_sf(xlim = c(-130, 40), ylim = c(23, 70)) +
  #     scale_fill_gradient(low = "#FFD7DA", high = "#ff5964",
  #                         na.value = "gray95", name = "Native country") +
  #     scale_color_manual(values = c("College" = "#E69F00", "Employer" = "#009E73"),
  #                        name = "Event type") +
  #     scale_size_continuous(name = "Events per location",
  #                           breaks = c(1, 10, 25, 50, 100, 250),
  #                           range = c(1, 10)) +
  #     theme_minimal() +
  #     labs(
  #       title = "Geographic mobility of queried individuals (USA & Europe)",
  #       caption = "Query via: GoA-DB. Basemap: World 1938 André Ourednik (GPL 3).",
  #       x = NULL, y = NULL
  #     ) +
  #     theme(
  #       legend.position = "right",
  #       plot.title = element_text(face = "bold", size = 14),
  #       legend.text = element_text(size = 10)
  #     )
  # })
  
  # ------------------------------------------------------------------------------
  # Visualization: Taiwan Map
  # ------------------------------------------------------------------------------
  
  # output$barChart_region_taiwan <- renderPlot({
  #   req(stat_region_employer())
  #   
  #   ggplot() +
  #     geom_sf(data = world_1938, fill = "gray95", color = "gray95", size = 0.1) +
  #     geom_sf(data = taiwan_1946, fill = "grey90", color = "white", size = 0.2) +
  #     geom_sf(data = stat_region_employer(),
  #             aes(geometry = geometry, size = n, color = type),
  #             alpha = 0.6) +
  #     coord_sf(xlim = c(119.3, 122), ylim = c(21.5, 25.5)) +
  #     scale_color_manual(values = c("Employer" = "#009E73"),
  #                        name = "Event type") +
  #     scale_size_continuous(name = "Events per location",
  #                           breaks = c(1, 10, 25, 50, 100, 250),
  #                           range = c(1, 10)) +
  #     theme_minimal() +
  #     labs(
  #       title = "Geographic mobility of queried individuals (Taiwan)",
  #       caption = "Query via: GoA-DB. Basemap: Ministry of the Interior / Taiwan Atlas.",
  #       x = NULL, y = NULL
  #     ) +
  #     theme(
  #       legend.position = "right",
  #       plot.title = element_text(face = "bold", size = 14),
  #       legend.text = element_text(size = 10)
  #     )
  # })
  
  # ------------------------------------------------------------------------------
  # Visualization: Birth Year Line Plot
  # ------------------------------------------------------------------------------
  
  output$barChart_birthyear <- renderPlot({
    
    stat_birthyear <- filtered_persons() %>%
      select(birthyear,person_id) %>%
      unique() %>%
      mutate(birthyear = as.numeric(birthyear)) %>%  # ensure numeric
      count(birthyear) %>%
      arrange(desc(n)) %>%
      drop_na()
    
    stat_graduation <- filtered_degrees() %>%
      select(year_graduated,person_id) %>%
      unique() %>%
      mutate(year_graduated = as.numeric(year_graduated)) %>%  # ensure numeric
      count(year_graduated) %>%
      arrange(desc(n)) %>%
      drop_na()
    
    ggplot() +
      # Birthyear line
      geom_line(
        data = stat_birthyear,
        aes(x = birthyear, y = n, group = 1, color = "Birth year"),
        linewidth = 0.6
      ) +
      geom_point(
        data = stat_birthyear,
        aes(x = birthyear, y = n, color = "Birth year"),
        size = 2
      ) +
      
      # Graduation line (stroked / dashed)
      geom_line(
        data = stat_graduation,
        aes(x = year_graduated, y = n, group = 1, color = "Graduation year"),
        linewidth = 0.6,
        linetype = "longdash"
      ) +
      geom_point(
        data = stat_graduation,
        aes(x = year_graduated, y = n, color = "Graduation year"),
        size = 2,
        color = "#ff5964",  # explicitly set color
        shape = 1  # hollow circle for contrast
      ) +
      
      scale_x_continuous(
        name = "Year",
        breaks = function(x) round(scales::breaks_pretty(n = 5)(x)),  # ≈5 nice breaks
        labels = function(x) as.integer(x)  # ensures integer labels
      ) +
      
      scale_color_manual(
        name = "",
        values = c("Birth year" = "#ff5964", "Graduation year" = "#ff5964")
      ) +
      
      scale_y_continuous(name = "Count", labels = scales::label_comma()) +
      labs(
        title = "Years of birth and graduation of queried individuals",
        subtitle = "",
        caption = "Query via: GoA-DB"
      ) +
      theme_minimal() +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        plot.title = element_text(face = "bold", size = 14),
        legend.position = "top",
        legend.title = element_blank(),
        legend.text = element_text(size = 12)  # increase legend text size
      )
  })
  
  # ------------------------------------------------------------------------------
  # Visualization: Employer Activity Plot
  # ------------------------------------------------------------------------------
  
  output$barChart_activity <- renderPlot({

    # "Tokyo" / "Tokyo Prefecture" / "Tokyo City" all describe the same place;
    # strip the administrative-unit suffix so they collapse into one bar.
    normalise_place <- function(x) {
      x <- gsub("\\s*\\((Prefecture|City|Ward|District|County|Borough|Town|Village)\\)$", "", x, ignore.case = TRUE)
      x <- gsub("\\s+(Prefecture|City|Ward|District|County|Borough|Town|Village)$", "", x, ignore.case = TRUE)
      x <- trimws(x)
      x
    }

    stat_jobs <- filtered_jobs() %>%
      select(start_year, person_id, organization_id, location_id) %>%
      unique() %>%
      # left_join(records_career %>% select(location_id, organization_id), by = "organization_id") %>%
      left_join(records_locations %>% select(location_id, name_en), by = "location_id") %>%
      mutate(start_year = as.numeric(start_year),
             name_en = normalise_place(name_en)) %>%
      drop_na()

    top10 <- stat_jobs %>%
      count(name_en, sort = TRUE) %>%
      slice_head(n = 10) %>%
      pull(name_en)
    
    stat_jobs <- stat_jobs %>%
      filter(name_en %in% top10) %>%
      mutate(name_en = fct_rev(fct_infreq(name_en)))
    
    # ggridges plot
    ggplot(stat_jobs, aes(x = start_year, y = name_en)) +
      geom_density_ridges(scale = 3, alpha = 0.6, color = "#ff5964", fill = "#ff5964",
                          bandwidth = 1) +
      scale_x_continuous(limits = c(1880, NA),
                         breaks = seq(1890, 1940, by = 10)) +
      theme_minimal() +
      labs(
        x = "Job start year",
        y = "Place",
        title = "Most active places by job commencements, queried individuals only",
        caption = "Query via: GoA-DB"
      ) +
      theme(
        axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
        axis.text.y = element_text(angle = 45, hjust = 1, size = 12),
        plot.title = element_text(face = "bold", size = 14),
        legend.position = "none",
        legend.title = element_blank(),
        legend.text = element_text(size = 12)  # increase legend text size
      )
  })
  
  # ------------------------------------------------------------------------------
  # Visualization: Job Title Bars
  # ------------------------------------------------------------------------------
  
  hisco_labels <- c(
    "0" = "Professional/Technical", "1" = "Professional/Technical",
    "2" = "Administrative/Managerial", "3" = "Office/Admin Staff",
    "4" = "Sales", "5" = "Service",
    "6" = "Agricultural/Forestry/Fishing",
    "7" = "Production/Transport", "8" = "Production/Transport",
    "9" = "Production/Transport"
  )

  hisco_colors <- c(
    "Professional/Technical" = "#0072B2", "Administrative/Managerial" = "#E69F00",
    "Office/Admin Staff" = "#009E73", "Sales" = "#CC79A7", "Service" = "#56B4E9",
    "Agricultural/Forestry/Fishing" = "#8C6D31", "Production/Transport" = "#D55E00",
    "Unclassified" = "#BBBBBB"
  )

  output$barChart_jobs <- renderPlot({

    jobs <- filtered_jobs()
    has_hisco <- "hisco_major" %in% names(jobs)

    if (has_hisco) {
      stat_jobs <- jobs %>%
        select(job_title, person_id, hisco_major) %>%
        unique() %>%
        drop_na(job_title) %>%
        mutate(hisco_group = ifelse(!is.na(hisco_major) & hisco_major %in% names(hisco_labels),
                                    hisco_labels[hisco_major], "Unclassified")) %>%
        count(job_title, hisco_group) %>%
        group_by(job_title) %>%
        summarise(n = sum(n), hisco_group = first(hisco_group), .groups = "drop") %>%
        arrange(desc(n)) %>%
        slice(1:10)

      ggplot(stat_jobs, aes(x = reorder(job_title, n), y = n, fill = hisco_group)) +
        geom_col() +
        geom_text(aes(label = paste0(" ", job_title)), y = 0, hjust = 0, size = 6, color = "black") +
        coord_flip() +
        scale_y_continuous(labels = scales::label_comma()) +
        scale_fill_manual(values = hisco_colors, na.value = "#BBBBBB") +
        theme_minimal() +
        labs(title = "Top-10 job titles of queried individuals",
             x = NULL, y = "Count", fill = "HISCO Group", caption = "Query via: GoA-DB") +
        theme(
          axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
          axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          plot.title = element_text(face = "bold", size = 14),
          legend.text = element_text(size = 10)
        )
    } else {
      # Fallback: no HISCO data available
      stat_jobs <- jobs %>%
        select(job_title, person_id) %>%
        unique() %>%
        drop_na() %>%
        count(job_title) %>%
        arrange(desc(n)) %>%
        slice(1:10)

      ggplot(stat_jobs, aes(x = reorder(job_title, n), y = n)) +
        geom_bar(stat = "identity", fill = "#ff5964") +
        coord_flip() +
        scale_y_continuous(labels = scales::label_comma()) +
        geom_text(aes(label = paste0(" ", job_title)), y = 0, hjust = 0, color = "black", size = 6) +
        theme_minimal() +
        labs(title = "Top-10 job titles of queried individuals",
             x = NULL, y = "Count", caption = "Query via: GoA-DB") +
        theme(
          axis.text.x = element_text(angle = 45, hjust = 1, size = 12),
          axis.text.y = element_blank(), axis.ticks.y = element_blank(),
          plot.title = element_text(face = "bold", size = 14)
        )
    }

  })
  
  # ------------------------------------------------------------------------------
  # Visualization: ISIC Treemap
  # ------------------------------------------------------------------------------

  output$treemap_isic <- renderPlot({

    isic_labels <- c(
      "A" = "Agriculture, forestry, fishing",
      "B" = "Mining and quarrying",
      "C" = "Manufacturing",
      "D" = "Electricity, gas, steam, AC",
      "E" = "Water, waste management",
      "F" = "Construction",
      "G" = "Wholesale and retail trade",
      "H" = "Transportation and storage",
      "I" = "Accommodation and food",
      "J" = "Information and communication",
      "K" = "Financial and insurance",
      "L" = "Real estate",
      "M" = "Professional, scientific, technical",
      "N" = "Administrative and support",
      "O" = "Public administration, defence",
      "P" = "Education",
      "Q" = "Health and social work",
      "R" = "Arts, entertainment, recreation",
      "S" = "Other service activities",
      "U" = "Extraterritorial organizations"
    )

    isic_emojis <- c(
      "A" = "\U0001F33E",  # ear of rice
      "B" = "\U000026CF",  # pick
      "C" = "\U0001F3ED",  # factory
      "D" = "\U000026A1",  # high voltage
      "E" = "\U0001F4A7",  # droplet
      "F" = "\U0001F3D7",  # building construction
      "G" = "\U0001F6D2",  # shopping cart
      "H" = "\U0001F69A",  # delivery truck
      "I" = "\U0001F3E8",  # hotel
      "J" = "\U0001F4E1",  # satellite antenna
      "K" = "\U0001F3E6",  # bank
      "L" = "\U0001F3D8",  # houses
      "M" = "\U0001F52C",  # microscope
      "N" = "\U0001F4CB",  # clipboard
      "O" = "\U0001F3DB",  # classical building
      "P" = "\U0001F393",  # graduation cap
      "Q" = "\U0001F3E5",  # hospital
      "R" = "\U0001F3AD",  # performing arts
      "S" = "\U0001F6E0",  # hammer and wrench
      "U" = "\U0001F310"   # globe with meridians
    )

    # Get ISIC sections from filtered employers via organization_id
    jobs <- filtered_jobs()
    req(nrow(jobs) > 0)

    treemap_data <- jobs %>%
      select(organization_id, person_id) %>%
      unique() %>%
      left_join(records_organizations %>% select(organization_id, isic_section), by = "organization_id") %>%
      filter(!is.na(isic_section) & isic_section %in% names(isic_labels)) %>%
      count(isic_section, name = "n") %>%
      mutate(label = isic_labels[isic_section],
             emoji = isic_emojis[isic_section],
             pct = sprintf("%.1f%%", n / sum(n) * 100))

    req(nrow(treemap_data) > 0)

    ggplot(treemap_data, aes(area = n, fill = n, label = paste0(emoji, "\n", label, "\n", pct))) +
      geom_treemap(colour = "white", size = 2) +
      geom_treemap_text(colour = "white", place = "centre",
                        grow = FALSE, reflow = TRUE, size = 11) +
      scale_fill_gradient(low = "#FFD7DA", high = "#ff5964", trans = "log10",
                          name = "Count") +
      theme_minimal() +
      labs(
        title = "Industry distribution of queried individuals (by ISIC code)",
        caption = "Query via: GoA-DB"
      ) +
      theme(
        plot.title = element_text(face = "bold", size = 14),
        legend.position = "none"
      )
  })

  # ------------------------------------------------------------------------------
  # Data Export Logics
  # ------------------------------------------------------------------------------
  
  # Download as Excel
  output$download_excel_person <- downloadHandler(
    filename = function() {
      paste0("cerd_filter_", Sys.Date(), ".xlsx")
    },
    content = function(file) {
      write_xlsx(dataTablePerson(), path = file)
    }
  )
  
  # Download as CSV
  output$download_csv_person <- downloadHandler(
    filename = function() {
      paste0("cerd_filter_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(dataTablePerson(), file, row.names = FALSE, 
                fileEncoding = "windows-1252")
    }
  )
  
  # Download as Excel (employers)
  output$download_excel_col <- downloadHandler(
    filename = function() {
      paste0("cerd_filter_", Sys.Date(), ".xlsx")
    },
    content = function(file) {
      write_xlsx(dataTableCol(), path = file)
    }
  )
  
  # Download as CSV (employers)
  output$download_csv_col <- downloadHandler(
    filename = function() {
      paste0("cerd_filter_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(dataTableCol(), file, row.names = FALSE)
    }
  )
  
  # Download as Excel (employers)
  output$download_excel_pub <- downloadHandler(
    filename = function() {
      paste0("cerd_filter_", Sys.Date(), ".xlsx")
    },
    content = function(file) {
      write_xlsx(dataTablePub(), path = file)
    }
  )
  
  # Download as CSV (employers)
  output$download_csv_pub <- downloadHandler(
    filename = function() {
      paste0("cerd_filter_", Sys.Date(), ".csv")
    },
    content = function(file) {
      write.csv(dataTablePub(), file, row.names = FALSE)
    }
  )
  
  # ------------------------------------------------------------------------------
  # Network Graph UI
  # ------------------------------------------------------------------------------
  
  output$dynamic_ui_netzwerk <- renderUI({
    req(g_sub())
    
    # Compute centrality measures
    deg <- degree(g_sub())
    btw <- betweenness(g_sub())
    clo <- closeness(g_sub(), normalized = TRUE, mode = "all")
    eig <- eigen_centrality(g_sub())$vector
    
    # Build summary using $title
    tagList(
      h4("Network summary"),
      p(paste("Number of nodes:", format(vcount(g_sub()), big.mark = ","))),
      p(paste("Number of edges:", format(ecount(g_sub()), big.mark = ","))),
      h5("Top nodes by centrality"),
      
      # Degree centrality
      p("Degree:"),
      tags$ul(
        lapply(1:3, function(i) {
          node_name <- V(g_sub())$name[order(deg, decreasing = TRUE)][i]
          node_title <- network_nodes_shared() %>% filter(id == node_name) %>% pull(title)
          node_value <- round(deg[which(V(g_sub())$name == node_name)], 2)
          tags$li(HTML(paste0('<a href="#" class="network-focus-link" data-id="', node_name, '">', node_title, '</a> (', node_value, ')')))
        })
      ),

      # Betweenness centrality
      p("Betweenness:"),
      tags$ul(
        lapply(1:3, function(i) {
          node_name <- V(g_sub())$name[order(btw, decreasing = TRUE)][i]
          node_title <- network_nodes_shared() %>% filter(id == node_name) %>% pull(title)
          node_value <- round(btw[which(V(g_sub())$name == node_name)], 2)
          tags$li(HTML(paste0('<a href="#" class="network-focus-link" data-id="', node_name, '">', node_title, '</a> (', node_value, ')')))
        })
      ),

      # Closeness centrality
      p("Closeness:"),
      tags$ul(
        lapply(1:3, function(i) {
          node_name <- V(g_sub())$name[order(clo, decreasing = TRUE)][i]
          node_title <- network_nodes_shared() %>% filter(id == node_name) %>% pull(title)
          node_value <- round(clo[which(V(g_sub())$name == node_name)], 2)
          tags$li(HTML(paste0('<a href="#" class="network-focus-link" data-id="', node_name, '">', node_title, '</a> (', node_value, ')')))
        })
      ),

      # Eigenvector centrality
      p("Eigenvector:"),
      tags$ul(
        lapply(1:3, function(i) {
          node_name <- V(g_sub())$name[order(eig, decreasing = TRUE)][i]
          node_title <- network_nodes_shared() %>% filter(id == node_name) %>% pull(title)
          node_value <- round(eig[which(V(g_sub())$name == node_name)], 2)
          tags$li(HTML(paste0('<a href="#" class="network-focus-link" data-id="', node_name, '">', node_title, '</a> (', node_value, ')')))
        })
      )
    )
  })

  # --------------------------------------------------------------------------
  # Network: Node detail view on click
  # --------------------------------------------------------------------------
  output$network_node_detail <- renderUI({
    node_id <- input$clicked_node
    if (is.null(node_id) || node_id == "") {
      return(div(style = "text-align:center;", h4("Click on a node for further information.")))
    }

    # Parse category from the suffixed ID (e.g. "P1927_42_person", "O123_employer")
    parts <- strsplit(node_id, "_(?=[^_]+$)", perl = TRUE)[[1]]
    if (length(parts) < 2) return(NULL)
    raw_id <- parts[1]
    category <- parts[2]

    if (category == "person") {
      clicked_person <- records_persons %>% filter(person_id == raw_id)
      if (nrow(clicked_person) == 0) return(p("Person not found."))
      clicked_person <- clicked_person[1, ]
      location <- records_locations[records_locations$location_id == clicked_person$location_id, ]

      clicked_degrees <- records_education %>%
        filter(person_id == raw_id) %>%
        filter(!if_all(-person_id, is.na)) %>%
        left_join(records_organizations %>% select(organization_id, organization), by = "organization_id") %>%
        mutate(has_valid_org = !is.na(organization) & nzchar(organization)) %>%
        filter(has_valid_org | (!is.na(major_of_study) & nzchar(major_of_study) & major_of_study != "NULL")) %>%
        mutate(edu_level = case_when(
          grepl("小学|小學|尋常|Elementary School|Primary School", organization, ignore.case = TRUE) ~ 1L,
          grepl("中学|中學|中等|Middle School", organization, ignore.case = TRUE) &
            !grepl("大学|大學", organization, ignore.case = TRUE) ~ 2L,
          grepl("高等学校|高等學校|高校|高等女|師範|予科|予備|High School|Normal School|Preparatory", organization, ignore.case = TRUE) &
            !grepl("大学|大學", organization, ignore.case = TRUE) ~ 3L,
          grepl("大学|大學|専門|專門|College|University", organization, ignore.case = TRUE) ~ 4L,
          grepl("大学院|研究科|Graduate", organization, ignore.case = TRUE) ~ 5L,
          TRUE ~ 3L
        )) %>%
        arrange(year_graduated, edu_level)

      net_person_volumes <- records_persons_core %>%
        filter(person_id == raw_id) %>%
        pull(volume) %>% unlist()
      net_volume_year <- suppressWarnings(max(volume_year_lookup[net_person_volumes], na.rm = TRUE))
      if (is.infinite(net_volume_year)) net_volume_year <- NA_integer_
      clicked_jobs <- records_career %>%
        filter(person_id == raw_id) %>%
        filter(!if_all(-person_id, is.na)) %>%
        mutate(sort_year = case_when(
          !is.na(start_year) ~ start_year,
          current %in% TRUE  ~ net_volume_year,
          TRUE               ~ NA_integer_
        )) %>%
        arrange(sort_year)

      display_name <- paste(
        if (!is.na(clicked_person$name_family_latin)) clicked_person$name_family_latin else NULL,
        if (!is.na(clicked_person$name_given_latin)) clicked_person$name_given_latin else NULL,
        if (!is.na(clicked_person$name)) clicked_person$name else NULL,
        if (!is.na(clicked_person$gender) && clicked_person$gender != "unknown") {
          if (clicked_person$gender == "m") "\u2642"
          else if (clicked_person$gender == "f") "\u2640"
          else ""
        } else ""
      )

      tagList(
        div(
          style = "display: flex; justify-content: space-between; align-items: flex-start;",
          h4(style = "margin-top: 0; margin-bottom: 0;",
             HTML(paste0('<a href="#" class="nav-person-link" data-id="', raw_id, '">', display_name, '</a>'))),
          span(paste("ID #", clicked_person$person_id))
        ),

        {
          loc_text <- if (!is.na(clicked_person$location_id) && nrow(location) > 0) format_location(location) else ""
          has_loc <- nzchar(loc_text)
          if (!is.na(clicked_person$birthyear) || has_loc) {
            p(HTML(paste0(
              if (!is.na(clicked_person$birthyear)) paste("born", clicked_person$birthyear) else NULL,
              if (has_loc)
                paste0(if (!is.na(clicked_person$birthyear)) '<br>' else '',
                       'active in ',
                       '<a href="#" class="location-link" data-id="', clicked_person$location_id, '">',
                       loc_text, '</a>') else NULL
            )))
          } else NULL
        },

        if (nrow(records_ranks %>% filter(person_id == clicked_person$person_id)) > 0) {
          clicked_ranks <- records_ranks %>% filter(person_id == clicked_person$person_id)
          p(HTML("<b>Rank</b>:"), HTML(paste(clicked_ranks$rank, collapse = ", ")))
        } else NULL,

        if (!is.na(clicked_person$tax_amount) && nzchar(clicked_person$tax_amount)) {
          p(HTML(paste0("<b>Tax:</b> ", clicked_person$tax_amount, " yen",
                        if (!is.na(net_volume_year)) paste0(" (by ", net_volume_year, ")") else "")))
        } else NULL,

        if (!is.na(clicked_person$political_party) && nzchar(clicked_person$political_party)) {
          p(HTML(paste0("<b>Political affiliation:</b> ", clicked_person$political_party,
                        if (!is.na(net_volume_year)) paste0(" (by ", net_volume_year, ")") else "")))
        } else NULL,

        if (!is.na(clicked_person$religion)) {
          clicked_religion_rows <- records_religion %>%
            filter(person_id == clicked_person$person_id)
          religion_parts <- vapply(seq_len(nrow(clicked_religion_rows)), function(i) {
            r <- clicked_religion_rows$religion[i]
            sect <- religions_dict %>% filter(religion == r) %>% pull(sect)
            sect <- gsub(" \\(generic\\)", "", sect)
            label <- if (length(sect) == 0 || is.na(sect[1])) r else paste(sect[1], r)
            vols <- unlist(clicked_religion_rows$source_volume[i])
            yr <- suppressWarnings(max(volume_year_lookup[vols], na.rm = TRUE))
            if (is.finite(yr)) paste0(label, " (by ", yr, ")") else label
          }, character(1))
          p(HTML(paste0(
            "<b>Religious affiliation:</b> ",
            paste(religion_parts, collapse = ", ")
          )))
        } else NULL,

        # Education
        if (nrow(clicked_degrees) > 0) {
          render_edu <- function(degree) {
            org_link <- ""
            has_org <- isTRUE(degree$has_valid_org)
            if (has_org) org_link <- paste0('<a href="#" class="college-link" data-id="', degree$organization_id, '">', degree$organization, '</a>')
            detail_parts <- c()
            if (!is.na(degree$major_of_study) && nzchar(degree$major_of_study) && degree$major_of_study != "NULL") detail_parts <- c(detail_parts, degree$major_of_study)
            if (!is.na(degree$year_graduated)) detail_parts <- c(detail_parts, as.character(degree$year_graduated))
            detail_text <- paste(detail_parts, collapse = ", ")
            paste0("<li>", org_link,
                   if (has_org && nzchar(detail_text)) paste0(" (", detail_text, ")")
                   else if (!has_org && nzchar(detail_text)) detail_text else "", "</li>")
          }
          tagList(
            p(HTML("<b>Education</b>:")),
            HTML(paste0("<ul>", paste(sapply(1:nrow(clicked_degrees), function(i) render_edu(clicked_degrees[i, ])), collapse = ""), "</ul>"))
          )
        } else NULL,

        # Employments
        if (nrow(clicked_jobs) > 0) {
          render_job <- function(job) {
            is_current <- isTRUE(job$current)
            detail_parts <- c()
            if (!is.na(job$start_year)) {
              detail_parts <- c(detail_parts, as.character(job$start_year))
            } else if (is_current && !is.na(net_volume_year)) {
              detail_parts <- c(detail_parts, paste0("by ", net_volume_year))
            }
            if (is_current) detail_parts <- c(detail_parts, "current")
            detail_text <- paste(detail_parts, collapse = ", ")
            paste0("<li>",
                   if (!is.na(job$job_title)) gsub("\\|", " & ", job$job_title) else "",
                   if (!is.na(job$organization_id)) {
                     org_name <- records_organizations %>% filter(organization_id == job$organization_id) %>% pull(organization)
                     if (length(org_name) > 0) paste0(' at <a href="#" class="employer-link" data-id="', job$organization_id, '">', org_name, '</a>') else ""
                   } else "",
                   if (nzchar(detail_text)) paste0(" (", detail_text, ")") else "",
                   "</li>")
          }
          jobs_dated <- clicked_jobs %>% filter(!is.na(sort_year))
          jobs_undated <- clicked_jobs %>% filter(is.na(sort_year))
          tagList(
            p(HTML("<b>Employments</b>:")),
            if (nrow(jobs_dated) > 0) HTML(paste0("<ol style='margin-bottom:0;'>", paste(sapply(1:nrow(jobs_dated), function(i) render_job(jobs_dated[i, ])), collapse = ""), "</ol>")) else NULL,
            if (nrow(jobs_undated) > 0) HTML(paste0("<ul style='margin-top:0;'>", paste(sapply(1:nrow(jobs_undated), function(i) render_job(jobs_undated[i, ])), collapse = ""), "</ul>")) else NULL
          )
        } else NULL,

        format_person_sources(raw_id)
      )

    } else if (category == "employer") {
      this_org <- records_organizations %>% filter(organization_id == raw_id)
      if (nrow(this_org) == 0) return(p("Organization not found."))
      location <- records_locations[records_locations$location_id == this_org$location_id[1], ]

      parent_org <- if (!is.na(this_org$parent_organization_id[1])) {
        records_organizations %>% filter(organization_id == this_org$parent_organization_id[1])
      } else data.frame()

      n_hq <- filtered_jobs() %>% filter(organization_id == raw_id) %>%
        select(person_id) %>% distinct() %>% nrow()

      tagList(
        div(
          style = "display: flex; justify-content: space-between; align-items: flex-start;",
          h4(style = "margin-top: 0; margin-bottom: 0;",
             HTML(paste0('<a href="#" class="nav-employer-link" data-id="', raw_id, '">', this_org$organization[1], '</a>'))),
          span(paste("ID #", raw_id))
        ),

        if (!is.na(this_org$location_id[1]) && nrow(location) > 0) {
          p(HTML(paste0('<a href="#" class="location-link" data-id="', this_org$location_id[1], '">',
                        format_location(location), '</a>')))
        } else NULL,

        if (!is.na(this_org$isic_label[1]) && nzchar(this_org$isic_label[1])) {
          p(HTML(paste0("<b>Industry:</b> ", this_org$isic_label[1], " (ISIC ", this_org$isic_section[1], ")")))
        } else NULL,

        if (nrow(parent_org) > 0) {
          p(HTML(paste0('Part of <a href="#" class="employer-link" data-id="', parent_org$organization_id[1], '">',
                        parent_org$organization[1], '</a>')))
        } else NULL,

        if (n_hq > 0) p(HTML(paste0('<b>Employees:</b> ', n_hq))) else NULL
      )

    } else if (category == "college") {
      this_org <- records_organizations %>% filter(organization_id == raw_id)
      if (nrow(this_org) == 0) return(p("College not found."))
      location <- records_locations[records_locations$location_id == this_org$location_id[1], ]

      tagList(
        div(
          style = "display: flex; justify-content: space-between; align-items: flex-start;",
          h4(style = "margin-top: 0; margin-bottom: 0;",
             HTML(paste0('<a href="#" class="nav-college-link" data-id="', raw_id, '">', this_org$organization[1], '</a>'))),
          span(paste("ID #", raw_id))
        ),
        if (!is.na(this_org$location_id[1]) && nrow(location) > 0) {
          p(HTML(paste0('<a href="#" class="location-link" data-id="', this_org$location_id[1], '">',
                        format_location(location), '</a>')))
        } else NULL
      )

    } else if (category == "location") {
      loc <- records_locations %>% filter(location_id == raw_id)
      if (nrow(loc) == 0) return(p("Location not found."))

      tagList(
        h4(style = "margin-top: 0; margin-bottom: 0;", format_location(loc)),
        span(paste("ID #", raw_id))
      )

    } else {
      NULL
    }
  })

}

# ==============================================================================
# RUN APPLICATION
# ==============================================================================
shinyApp(ui = ui, server = server)
