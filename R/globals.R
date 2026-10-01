# Declare variables used in non-standard evaluation (ggplot aes, dplyr, with())
# to suppress R CMD check NOTE about no visible binding for global variable
utils::globalVariables(c(
  # ggplot aes mappings
  "value", "group", "density", "variable", "outcome",
  # with() / dplyr column references
  "cont_var1", "cont_var2", "cont_var3", "cont_var4", "cont_var5",
  "cont_var6", "cont_var7", "bin_var1", "bin_var2", "bin_var3",
  "outcome0", "outcome1"
))