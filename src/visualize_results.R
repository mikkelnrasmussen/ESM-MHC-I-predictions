# Load necessary libraries
library(ggplot2)
library(dplyr)
library(stringr)
library(here)
library(tidyr)
library(ggpubr)
library(RColorBrewer)

# Specify the directory
DIR_PATH <- "results"

# Define the encoding schemes
encodings <- c("ESM", "sparse", "blosum")

# Define the alleles
alleles <- c(
    "A0101", "A0201", "A0202", "A0203", "A0206", "A0301", "A1101", "A2402", 
    "A2403", "A2601", "A2902", "A3001", "A3002", "A3101", "A3301", "A6801", 
    "A6802", "A6901", "B0702", "B0801", "B1501", "B1801", "B2705", "B3501", 
    "B4001", "B4002", "B4402", "B4403", "B4501", "B5101", "B5301", "B5401", 
    "B5701", "B5801"
)

# Initialize an empty data frame to store the results
results <- data.frame()

# Iterate over each encoding scheme and allele
for (e in encodings) {
  for (a in alleles) {
    # Initialize variables to store the total AUC and MCC values and the
    # count of files
    total_AUC <- 0
    total_MCC <- 0
    file_count <- 0

    # Iterate over each file index
    for(i in 0:4){
      # Formulate the specific file path
      SPECIFIC_FILE_PATH <- paste0(
        DIR_PATH, "/", a, "/", e, "_with_sigmoid/best_models/",
         a, "_", e, "_", i, "_performance_results.out"
        )

      # Check if the file exists
      if (file.exists(here(SPECIFIC_FILE_PATH))) {
        # Read the file content
        file_content <- readLines(here(SPECIFIC_FILE_PATH))

        # Extract AUC and MCC values
        AUC <- as.numeric(str_extract(file_content[2], "\\d+\\.\\d+"))
        MCC <- as.numeric(str_extract(file_content[3], "\\d+\\.\\d+"))

        # Add the values to the totals
        total_AUC <- total_AUC + AUC
        total_MCC <- total_MCC + MCC

        # Increment the file count
        file_count <- file_count + 1
      }
    }

    # Check if there was at least one file
    if(file_count > 0) {
      # Calculate the average AUC and MCC values
      avg_AUC <- total_AUC / file_count
      avg_MCC <- total_MCC / file_count

      # Add the results to the data frame along with the corresponding
      # encoding scheme and allele
      temp_df <- data.frame(
        Encoding = e, Allele = a, AUC = avg_AUC, MCC = avg_MCC
        )
      results <- rbind(results, temp_df)
    }
  }
}

# Define a data frame to store the data sizes
data_sizes <- data.frame()
for (a in alleles) {
  for (i in 0:4) {
    #Formulate the specific file path
    eval_file <- paste0(
      "data/processed_data/", a, "/", paste0("c00", i)
    )
    train_file <- paste0(
      "data/processed_data/", a, "/", paste0("f00", i)
    )

    # Initialize row counters
    eval_rows <- 0
    train_rows <- 0

    # Check if eval file is not empty before reading
    if (file.size(eval_file) > 0) {
      # Read the number of rows in the evaluation file
      eval_data <- read.csv(here(eval_file), header = FALSE)
      eval_rows <- nrow(eval_data)
    }

    # Check if train file is not empty before reading
    if (file.size(train_file) > 0) {
      # Read the number of rows in the training file
      train_data <- read.csv(here(train_file), header = FALSE)
      train_rows <- nrow(train_data)
    }

    # Add the number of data points in train and eval files
    total_rows <- eval_rows + train_rows
    temp_df <- data.frame(
      Allele = a, Data_Size = total_rows
    )
    data_sizes <- rbind(data_sizes, temp_df)
  }
}

# Check the data sizes are the same across different folds
data_sizes_unique <- data_sizes %>%
  group_by(Allele) %>%
  dplyr::reframe(
    Data_Size = unique(Data_Size)
  )
data_sizes_unique[data_sizes_unique %>% select(Allele) %>% duplicated(), ]

# Define a color palette
# Change the number based on your number of groups (encoding schemes)
#color_palette <- brewer.pal(3, "Set2") 
color_palette <- c("#FF2400", "#1A5FA1", "#50C878")

# Plot the results for AUC
# Define the y-axis breaks
y_breaks <- seq(from = round(min(results$AUC, na.rm = TRUE)),
                to = max(results$AUC, na.rm = TRUE), 
                by = 0.05) # Adjust the "by" argument to change the distance between ticks
# Define a mapping from original encoding names to new encoding names
new_names <- c("blosum" = "BLOSUM62", "ESM" = "ESM-1b", "sparse" = "Soft-sparse")

# Set the factor levels for the Encoding column
results$Encoding <- factor(results$Encoding, levels = c("sparse", "blosum", "ESM"))

# Extract results for each encoding scheme
results_ESM <- results %>%
  filter(Encoding == "ESM") %>%
  arrange(Allele) %>%
  drop_na() %>%
  pull(AUC)

results_blosum <- results %>%
  filter(Encoding == "blosum") %>%
  arrange(Allele) %>%
  drop_na() %>%
  pull(AUC)

results_sparse <- results %>%
  filter(Encoding == "sparse") %>%
  arrange(Allele) %>%
  drop_na() %>%
  pull(AUC)

# Find difference and set negative values to 0 and positive values to 1
results_ESM_blosum <- results_blosum - results_ESM
results_ESM_blosum[results_ESM_blosum < 0] <- 0
results_ESM_blosum[results_ESM_blosum > 0] <- 1

results_sparse_ESM <- results_sparse - results_ESM
results_sparse_ESM[results_sparse_ESM < 0] <- 0
results_sparse_ESM[results_sparse_ESM > 0] <- 1

results_sparse_blosum <- results_blosum - results_sparse
results_sparse_blosum[results_sparse_blosum < 0] <- 0
results_sparse_blosum[results_sparse_blosum > 0] <- 1

# ESM vs blosum
p_ESM_blosum <- binom.test(
  sum(results_ESM_blosum),
  length(results_ESM_blosum),
  p = 0.5,
  alternative = "greater"
)$p.value
p_ESM_blosum
t.test(
  results_blosum,
  results_ESM,
  paired = TRUE,
  alternative = "greater"
)
wilcox.test(
  results_blosum,
  results_ESM,
  paired = TRUE,
  alternative = "greater"
)

p_sparse_ESM <- binom.test(
  sum(results_sparse_ESM),
  length(results_sparse_ESM),
  p = 0.5,
  alternative = "greater"
)$p.value
p_sparse_ESM
t.test(
  results_sparse,
  results_ESM,
  paired = TRUE,
  alternative = "greater"
)
wilcox.test(
  results_sparse,
  results_ESM,
  paired = TRUE,
  alternative = "greater"
)

p_sparse_blosum <- binom.test(
  sum(results_sparse_blosum),
  length(results_sparse_blosum),
  p = 0.5,
  alternative = "greater"
)$p.value
p_sparse_blosum
t.test(
  results_blosum,
  results_sparse,
  paired = TRUE,
  alternative = "greater"
)
wilcox.test(
  results_blosum,
  results_sparse,
  paired = TRUE,
  alternative = "greater"
)

results <- results %>% tidyr::drop_na()

boxplot <- results %>%
unique() %>%
ggplot(aes(x = Encoding, y = AUC, fill = Encoding)) +
  geom_boxplot(alpha = 0.5) +
  scale_y_continuous(breaks = y_breaks) +
  scale_x_discrete(labels = new_names) +
  scale_fill_manual(values = color_palette) +
  geom_point(
    position = position_jitter(width = 0.4, seed = 1),
    color = "black") +
  geom_text(
    aes(label=Allele),
    size = 4,
    position = position_jitter(width = 0.4, seed = 1), 
    vjust = -1,
    check_overlap = TRUE) +
  theme_minimal() +
  labs(
    #title = "Average Performance Results - AUC",
    x = "Encoding Scheme",
    y = "AUC") +
  theme(legend.position = "none",
    text = element_text(size = 20), # Global text size
    axis.text = element_text(size = 16), # Axis text size
    axis.title = element_text(size = 18), # Axis title size
    plot.title = element_text(size = 22)) # Plot title size
# Add the p-values and brackets
boxplot <- boxplot +
  stat_compare_means(comparisons = list(c("blosum", "ESM")), 
                     method = "wilcox.test", 
                     paired = TRUE, 
                     method.args = list(alternative = "greater"),
                     label.x.npc = "center", # Position label in the center of comparison
                     label = "p.format", # Add the p-value
                     label.y = max(results$AUC, na.rm = TRUE) + 0.015) + 
  stat_compare_means(comparisons = list(c("sparse", "ESM")),
                     method = "wilcox.test",
                     paired = TRUE,
                     method.args = list(alternative = "greater"),
                     label.x.npc = "center", # Position label in the center of comparison
                     label = "p.format", # Add the p-value
                     label.y = max(results$AUC, na.rm = TRUE) + 0.05) + 
  stat_compare_means(comparisons = list(c("blosum", "sparse")),
                     method = "wilcox.test",
                     paired = TRUE,
                     method.args = list(alternative = "greater"),
                     label.x.npc = "center", # Position label in the center of comparison
                     label = "p.format", # Add the p-value
                     label.y = max(results$AUC, na.rm = TRUE) + 0.10)
boxplot
ggsave("plots/average_AUC_performannce_with_sigmoid.png", width = 9, height = 8)

# Plot the results for MCC
results %>%
drop_na() %>%
unique() %>%
ggplot(aes(x = Encoding, y = MCC, fill = Encoding)) +
  geom_boxplot(alpha = 0.5) +
  scale_fill_manual(values=color_palette) +
  scale_y_continuous(breaks = y_breaks) +
  geom_point(
    position = position_jitter(width = 0.3, seed = 1),
    color = "black") +
  geom_text(
    aes(label=Allele),
    size = 3,
    position = position_jitter(width = 0.3, seed = 1), 
    vjust = -1,
    check_overlap = TRUE) +
  theme_minimal() +
  labs(
    title = "Average Performance Results - MCC", 
    x = "Encoding Scheme",
    y = "MCC")


# Merge results and data_sizes
merged_data <- merge(results, data_sizes, by = "Allele")

# Create the plot
merged_data %>%
  drop_na() %>%
  ggplot(aes(x = Data_Size, y = AUC, color = Encoding, shape = Encoding)) +
  geom_point(size = 3) + # Increase the size here to make symbols bolder
  labs(
    x = "Data Size",
    y = "AUC",
    color = "Encoding Scheme",
    shape = "Encoding Scheme"
  ) +
  theme_minimal() +
  scale_y_continuous(breaks = y_breaks) +
  scale_color_manual(values = color_palette, 
                     labels = c("Soft-sparse", "BLOSUM62", "ESM-1b")) + # Assign the new names to labels
  scale_shape_manual(values=c(15, 16, 17), # Solid shapes
                     labels = c("Soft-sparse", "BLOSUM62", "ESM-1b")) + # Assign the new names to labels
  theme(
    legend.position = c(0.75, 0.2), # Bottom right corner
    text = element_text(size = 20), # Global text size
    axis.text = element_text(size = 16), # Axis text size
    axis.title = element_text(size = 18), # Axis title size
    plot.title = element_text(size = 22), # Plot title size
    legend.background = element_blank(), # Removes legend box background
    legend.box.background = element_rect(colour = "black") # Adds a black border to the legend box
  )
ggsave("plots/data_size_vs_AUC_with_sigmoid.png", width = 9, height = 7)
#facet_wrap(~Encoding, ncol = 1)


# Reshape the data to wide format
library(tidyr)

results_wide <- results %>% 
  dplyr::select(Encoding, Allele, AUC) %>%
  spread(Encoding, AUC)

# Load the necessary library
library(xtable)

# Convert the data frame to a LaTeX table
latex_table <- xtable(results_wide)

print(latex_table, include.rownames = FALSE)
