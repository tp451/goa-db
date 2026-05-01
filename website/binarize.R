# ==============================================================================
# Pre-process all data for the Shiny app into a single RDS file
# ==============================================================================
# Run this script once (or after data changes) to create app_data.rds.
# The Shiny app then loads that single file instead of parsing GeoJSON/JSONL/CSV.
#
# Usage:  Rscript binarize.R
# ==============================================================================

library(tidyverse)
library(sf)
library(jsonlite)

cat("Loading GeoJSON files...\n")

map_china <- read_sf("china_1928_east.geojson") %>% st_make_valid() %>% mutate(country = "zh")
map_taiwan <- read_sf("taiwan_1946.geojson") %>% st_make_valid() %>%
  select(geometry, COUNTYENG) %>% rename(province = COUNTYENG) %>% mutate(country = "tw")
map_japan <- read_sf("japan_hijmans.geojson") %>% st_make_valid() %>%
  select(NAME_1, geometry) %>% rename(province = NAME_1) %>% mutate(country = "ja")
map_korea <- read_sf("korea_imperial.geojson") %>% st_make_valid() %>% mutate(country = "kr")

map_asia_outer <- rbind(map_china, map_taiwan, map_korea)
map_asia <- rbind(map_asia_outer, map_japan)

# Province -> country lookup from map shapefiles
province_country <- c(
  setNames(rep("ja", nrow(map_japan)), map_japan$province),
  setNames(rep("tw", nrow(map_taiwan)), map_taiwan$province),
  setNames(rep("zh", nrow(map_china)), map_china$province),
  setNames(rep("kr", nrow(map_korea)), trimws(map_korea$province)),
  setNames(rep("zh", 8), c("Gansu", "Ningxia", "Qinghai", "Sichuan", "Tibet", "Xikang", "Xinjiang", "Yunnan"))
)
manchukuo_provinces <- c("Heilongjiang", "Jilin", "Liaoning", "Jehol")

cat("Loading CSV files...\n")

religions_dict <- read_csv("religions.csv", show_col_types = FALSE)

religion_category_map <- religions_dict %>%
  filter(!is.na(belongs_to) & nzchar(belongs_to)) %>%
  mutate(picker_cat = case_when(
    belongs_to == "Buddhism" & school == "Pure Land" ~ "pureland",
    belongs_to == "Buddhism" & school == "Zen" ~ "zen",
    belongs_to == "Buddhism" & school == "Nichiren" ~ "nichiren",
    belongs_to == "Buddhism" ~ "buddhism",
    belongs_to == "Shint\u014d" ~ "shinto",
    belongs_to == "Christianity" ~ "christianity",
    belongs_to == "Confucianism" ~ "confucianism",
    belongs_to == "Tenriky\u014d" ~ "tenrikyo",
    TRUE ~ "other"
  ))

hobbies_dict <- read_csv("hobbies.csv", show_col_types = FALSE)
relations_dict <- read_csv("relations.csv", show_col_types = FALSE)

cat("Loading JSONL files...\n")

# --- Locations ---
records_locations <- stream_in(file("data/locations.jsonl"), simplifyVector = TRUE)

coords <- as.matrix(records_locations[, c("longitude", "latitude")])
geom <- st_sfc(
  lapply(seq_len(nrow(coords)), function(i) st_point(coords[i, ])),
  crs = 4326
)
records_locations$geometry <- geom
records_locations <- st_as_sf(records_locations)

records_locations <- records_locations %>%
  mutate(location_name = ifelse(is.na(admin2), admin1, paste(admin1, admin2)))
records_locations <- records_locations %>% select(-name)
records_locations$country <- province_country[records_locations$province]

# Helper: format location as "English City, English District ..."
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

records_locations$location_display <- sapply(1:nrow(records_locations),
  function(i) format_location(records_locations[i, ]))
records_locations$location_display_short <- sapply(1:nrow(records_locations), function(i) {
  row <- records_locations[i, ]
  en_str <- if (!is.na(row$admin1_en) && nzchar(row$admin1_en)) row$admin1_en else ""
  ja_str <- if (!is.na(row$admin1) && nzchar(row$admin1)) row$admin1 else ""
  if (nzchar(en_str) && nzchar(ja_str)) paste(en_str, ja_str)
  else if (nzchar(ja_str)) ja_str
  else en_str
})

# --- Person core ---
records_persons_core <- stream_in(file("data/person_core.jsonl"), simplifyVector = TRUE)

# --- Person appearances (volume + page in source) ---
records_appearances <- stream_in(file("data/person_appearances.jsonl"), simplifyVector = TRUE) %>%
  select(person_id, volume, source_page) %>%
  mutate(page = suppressWarnings(as.integer(sub("^[^_]*_0*", "", source_page)))) %>%
  distinct()

# --- Ranks ---
records_ranks <- stream_in(file("data/person_ranks.jsonl"), simplifyVector = TRUE)

# --- Hobbies ---
records_hobbies <- stream_in(file("data/person_hobbies.jsonl"), simplifyVector = TRUE) %>%
  left_join(hobbies_dict, by = "hobby") %>%
  mutate(hobby = ifelse(is.na(hobby_en), hobby, paste(hobby_en, hobby)))

# --- Religion ---
records_religion <- stream_in(file("data/person_religions.jsonl"), simplifyVector = TRUE)

# --- Political parties ---
records_political_parties <- if (file.exists("data/person_political_parties.jsonl")) {
  stream_in(file("data/person_political_parties.jsonl"), simplifyVector = TRUE)
} else {
  data.frame(person_id = character(), political_party = character(), stringsAsFactors = FALSE)
}

# --- Persons (joined) ---
records_persons <- records_persons_core %>%
  filter(!(domain == "ja" & (is.na(name_family_latin) | !nzchar(name_family_latin)))) %>%
  left_join(records_religion, by = "person_id", relationship = "many-to-many") %>%
  left_join(records_political_parties, by = "person_id", relationship = "many-to-many") %>%
  left_join(records_locations %>% st_drop_geometry(), by = "location_id")

# --- Family members ---
records_family <- stream_in(file("data/person_family_members.jsonl"), simplifyVector = TRUE) %>%
  mutate(order = case_when(
    grepl("祖先|先祖|遠祖|元祖|高祖|玄祖|曾祖|祖父|祖母|先代|先々", relation) ~ 0L,
    grepl("^祖$", relation) ~ 0L,
    grepl("伯父|伯母|叔父|叔母|舅|姑父|姑母|姑", relation) ~ 2L,
    grepl("父|母", relation) ~ 1L,
    grepl("妻|夫$|夫人|夫君|婿|嫁", relation) ~ 4L,
    grepl("従兄|従弟|従姉|従妹|從兄|從弟|從姉|從妹", relation) ~ 6L,
    grepl("兄|姉|弟|妹", relation) ~ 3L,
    grepl("子|男|女|嗣|娘", relation) ~ 5L,
    grepl("甥|姪", relation) ~ 7L,
    grepl("孫|曾孫|玄孫", relation) ~ 8L,
    TRUE ~ 9L
  )) %>%
  mutate(order2 = ifelse(grepl("長", relation), 1, 3)) %>%
  mutate(order2 = ifelse(grepl("次", relation), 2, order2)) %>%
  mutate(order3 = ifelse(grepl("一", relation), 1, 6)) %>%
  mutate(order3 = ifelse(grepl("二", relation), 2, order3)) %>%
  mutate(order3 = ifelse(grepl("三", relation), 3, order3)) %>%
  mutate(order3 = ifelse(grepl("四", relation), 4, order3)) %>%
  mutate(order3 = ifelse(grepl("五", relation), 5, order3)) %>%
  left_join(relations_dict, by = "relation") %>%
  mutate(relation = ifelse(is.na(relation_en), relation, paste(relation_en, relation)))

# --- Family education ---
records_family_education <- tryCatch(
  stream_in(file("data/person_family_education.jsonl"), simplifyVector = TRUE),
  error = function(e) data.frame(person_id = character(), relation_id = character(),
                                  organization_id = character(), major_of_study = character(),
                                  year_graduated = integer(), stringsAsFactors = FALSE)
)

# --- Family career ---
records_family_career <- tryCatch(
  stream_in(file("data/person_family_career.jsonl"), simplifyVector = TRUE) %>%
    mutate(job_title = ifelse(is.na(job_title_en), job_title, paste(job_title_en, job_title))),
  error = function(e) data.frame(person_id = character(), relation_id = character(),
                                  organization_id = character(), job_title = character(),
                                  start_year = integer(), stringsAsFactors = FALSE)
)

# --- Organizations ---
records_organizations <- stream_in(file("data/organizations.jsonl"), simplifyVector = TRUE) %>%
  rename(organization = name, org_name_en = name_en) %>%
  left_join(records_locations %>% st_drop_geometry(), by = "location_id") %>%
  mutate(organization = ifelse(is.na(org_name_en), organization, paste(org_name_en, organization))) %>%
  mutate(isic_label = if ("isic_label" %in% names(.)) isic_label else NA_character_) %>%
  mutate(isic_label = recode(isic_label,
    "Accommodation and food service activities" = "Accommodation and food",
    "Activities of extraterritorial organizations and bodies" = "Extraterritorial organizations",
    "Activities of households as employers" = "Households as employers",
    "Administrative and support service activities" = "Administrative and support",
    "Electricity, gas, steam and air conditioning supply" = "Electricity, gas, steam and AC",
    "Financial and insurance activities" = "Financial and insurance",
    "Human health and social work activities" = "Health and social work",
    "Other service activities" = "Other services",
    "Professional, scientific and technical activities" = "Professional, scientific, technical",
    "Public administration and defence; compulsory social security" = "Public administration and defence",
    "Real estate activities" = "Real estate",
    "Water supply; sewerage, waste management and remediation activities" = "Water and waste management",
    "Wholesale and retail trade; repair of motor vehicles and motorcycles" = "Wholesale and retail trade"
  ))

# --- Education ---
records_education <- stream_in(file("data/person_education.jsonl"), simplifyVector = TRUE)

# --- Career ---
records_career <- stream_in(file("data/person_career.jsonl"), simplifyVector = TRUE) %>%
  mutate(current = as.logical(current)) %>%
  mutate(job_title = ifelse(is.na(job_title_en), job_title, paste(job_title_en, job_title)))

# --- Person career countries ---
loc_country <- records_locations %>% st_drop_geometry() %>% select(location_id, country, province)
person_career_countries <- records_career %>%
  select(person_id, location_id) %>%
  filter(!is.na(location_id)) %>%
  distinct() %>%
  left_join(loc_country, by = "location_id") %>%
  filter(!is.na(country)) %>%
  select(person_id, country, province) %>%
  distinct()

# ==============================================================================
# Save everything as a single RDS file
# ==============================================================================

cat("Saving app_data.rds...\n")

saveRDS(list(
  # Map data (sf objects)
  map_china = map_china,
  map_taiwan = map_taiwan,
  map_japan = map_japan,
  map_korea = map_korea,
  map_asia_outer = map_asia_outer,
  map_asia = map_asia,

  # Lookups
  province_country = province_country,
  manchukuo_provinces = manchukuo_provinces,

  # Dictionaries
  religions_dict = religions_dict,
  religion_category_map = religion_category_map,
  hobbies_dict = hobbies_dict,
  relations_dict = relations_dict,

  # Records (sf / data.frame)
  records_locations = records_locations,
  records_persons_core = records_persons_core,
  records_persons = records_persons,
  records_appearances = records_appearances,
  records_ranks = records_ranks,
  records_hobbies = records_hobbies,
  records_religion = records_religion,
  records_political_parties = records_political_parties,
  records_family = records_family,
  records_family_education = records_family_education,
  records_family_career = records_family_career,
  records_organizations = records_organizations,
  records_education = records_education,
  records_career = records_career,
  person_career_countries = person_career_countries
), "app_data.rds")

cat("Done. app_data.rds written.\n")
